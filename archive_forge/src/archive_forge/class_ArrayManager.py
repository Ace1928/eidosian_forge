from __future__ import annotations
import itertools
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.take import take_1d
from pandas.core.arrays import (
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.indexes.api import (
from pandas.core.indexes.base import get_values_for_csv
from pandas.core.internals.base import (
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import make_na_array
class ArrayManager(BaseArrayManager):

    @property
    def ndim(self) -> Literal[2]:
        return 2

    def __init__(self, arrays: list[np.ndarray | ExtensionArray], axes: list[Index], verify_integrity: bool=True) -> None:
        self._axes = axes
        self.arrays = arrays
        if verify_integrity:
            self._axes = [ensure_index(ax) for ax in axes]
            arrays = [extract_pandas_array(x, None, 1)[0] for x in arrays]
            self.arrays = [maybe_coerce_values(arr) for arr in arrays]
            self._verify_integrity()

    def _verify_integrity(self) -> None:
        n_rows, n_columns = self.shape_proper
        if not len(self.arrays) == n_columns:
            raise ValueError(f'Number of passed arrays must equal the size of the column Index: {len(self.arrays)} arrays vs {n_columns} columns.')
        for arr in self.arrays:
            if not len(arr) == n_rows:
                raise ValueError(f'Passed arrays should have the same length as the rows Index: {len(arr)} vs {n_rows} rows')
            if not isinstance(arr, (np.ndarray, ExtensionArray)):
                raise ValueError(f'Passed arrays should be np.ndarray or ExtensionArray instances, got {type(arr)} instead')
            if not arr.ndim == 1:
                raise ValueError(f'Passed arrays should be 1-dimensional, got array with {arr.ndim} dimensions instead.')

    def fast_xs(self, loc: int) -> SingleArrayManager:
        """
        Return the array corresponding to `frame.iloc[loc]`.

        Parameters
        ----------
        loc : int

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        dtype = interleaved_dtype([arr.dtype for arr in self.arrays])
        values = [arr[loc] for arr in self.arrays]
        if isinstance(dtype, ExtensionDtype):
            result = dtype.construct_array_type()._from_sequence(values, dtype=dtype)
        elif is_datetime64_ns_dtype(dtype):
            result = DatetimeArray._from_sequence(values, dtype=dtype)._ndarray
        elif is_timedelta64_ns_dtype(dtype):
            result = TimedeltaArray._from_sequence(values, dtype=dtype)._ndarray
        else:
            result = np.array(values, dtype=dtype)
        return SingleArrayManager([result], [self._axes[1]])

    def get_slice(self, slobj: slice, axis: AxisInt=0) -> ArrayManager:
        axis = self._normalize_axis(axis)
        if axis == 0:
            arrays = [arr[slobj] for arr in self.arrays]
        elif axis == 1:
            arrays = self.arrays[slobj]
        new_axes = list(self._axes)
        new_axes[axis] = new_axes[axis]._getitem_slice(slobj)
        return type(self)(arrays, new_axes, verify_integrity=False)

    def iget(self, i: int) -> SingleArrayManager:
        """
        Return the data as a SingleArrayManager.
        """
        values = self.arrays[i]
        return SingleArrayManager([values], [self._axes[0]])

    def iget_values(self, i: int) -> ArrayLike:
        """
        Return the data for column i as the values (ndarray or ExtensionArray).
        """
        return self.arrays[i]

    @property
    def column_arrays(self) -> list[ArrayLike]:
        """
        Used in the JSON C code to access column arrays.
        """
        return [np.asarray(arr) for arr in self.arrays]

    def iset(self, loc: int | slice | np.ndarray, value: ArrayLike, inplace: bool=False, refs=None) -> None:
        """
        Set new column(s).

        This changes the ArrayManager in-place, but replaces (an) existing
        column(s), not changing column values in-place).

        Parameters
        ----------
        loc : integer, slice or boolean mask
            Positional location (already bounds checked)
        value : np.ndarray or ExtensionArray
        inplace : bool, default False
            Whether overwrite existing array as opposed to replacing it.
        """
        if lib.is_integer(loc):
            if isinstance(value, np.ndarray) and value.ndim == 2:
                assert value.shape[1] == 1
                value = value[:, 0]
            value = maybe_coerce_values(value)
            assert isinstance(value, (np.ndarray, ExtensionArray))
            assert value.ndim == 1
            assert len(value) == len(self._axes[0])
            self.arrays[loc] = value
            return
        elif isinstance(loc, slice):
            indices: range | np.ndarray = range(loc.start if loc.start is not None else 0, loc.stop if loc.stop is not None else self.shape_proper[1], loc.step if loc.step is not None else 1)
        else:
            assert isinstance(loc, np.ndarray)
            assert loc.dtype == 'bool'
            indices = np.nonzero(loc)[0]
        assert value.ndim == 2
        assert value.shape[0] == len(self._axes[0])
        for value_idx, mgr_idx in enumerate(indices):
            value_arr = value[:, value_idx]
            self.arrays[mgr_idx] = value_arr
        return

    def column_setitem(self, loc: int, idx: int | slice | np.ndarray, value, inplace_only: bool=False) -> None:
        """
        Set values ("setitem") into a single column (not setting the full column).

        This is a method on the ArrayManager level, to avoid creating an
        intermediate Series at the DataFrame level (`s = df[loc]; s[idx] = value`)
        """
        if not is_integer(loc):
            raise TypeError('The column index should be an integer')
        arr = self.arrays[loc]
        mgr = SingleArrayManager([arr], [self._axes[0]])
        if inplace_only:
            mgr.setitem_inplace(idx, value)
        else:
            new_mgr = mgr.setitem((idx,), value)
            self.arrays[loc] = new_mgr.arrays[0]

    def insert(self, loc: int, item: Hashable, value: ArrayLike, refs=None) -> None:
        """
        Insert item at selected position.

        Parameters
        ----------
        loc : int
        item : hashable
        value : np.ndarray or ExtensionArray
        """
        new_axis = self.items.insert(loc, item)
        value = extract_array(value, extract_numpy=True)
        if value.ndim == 2:
            if value.shape[0] == 1:
                value = value[0, :]
            else:
                raise ValueError(f'Expected a 1D array, got an array with shape {value.shape}')
        value = maybe_coerce_values(value)
        arrays = self.arrays.copy()
        arrays.insert(loc, value)
        self.arrays = arrays
        self._axes[1] = new_axis

    def idelete(self, indexer) -> ArrayManager:
        """
        Delete selected locations in-place (new block and array, same BlockManager)
        """
        to_keep = np.ones(self.shape[0], dtype=np.bool_)
        to_keep[indexer] = False
        self.arrays = [self.arrays[i] for i in np.nonzero(to_keep)[0]]
        self._axes = [self._axes[0], self._axes[1][to_keep]]
        return self

    def grouped_reduce(self, func: Callable) -> Self:
        """
        Apply grouped reduction function columnwise, returning a new ArrayManager.

        Parameters
        ----------
        func : grouped reduction function

        Returns
        -------
        ArrayManager
        """
        result_arrays: list[np.ndarray] = []
        result_indices: list[int] = []
        for i, arr in enumerate(self.arrays):
            arr = ensure_block_shape(arr, ndim=2)
            res = func(arr)
            if res.ndim == 2:
                assert res.shape[0] == 1
                res = res[0]
            result_arrays.append(res)
            result_indices.append(i)
        if len(result_arrays) == 0:
            nrows = 0
        else:
            nrows = result_arrays[0].shape[0]
        index = Index(range(nrows))
        columns = self.items
        return type(self)(result_arrays, [index, columns])

    def reduce(self, func: Callable) -> Self:
        """
        Apply reduction function column-wise, returning a single-row ArrayManager.

        Parameters
        ----------
        func : reduction function

        Returns
        -------
        ArrayManager
        """
        result_arrays: list[np.ndarray] = []
        for i, arr in enumerate(self.arrays):
            res = func(arr, axis=0)
            dtype = arr.dtype if res is NaT else None
            result_arrays.append(sanitize_array([res], None, dtype=dtype))
        index = Index._simple_new(np.array([None], dtype=object))
        columns = self.items
        new_mgr = type(self)(result_arrays, [index, columns])
        return new_mgr

    def operate_blockwise(self, other: ArrayManager, array_op) -> ArrayManager:
        """
        Apply array_op blockwise with another (aligned) BlockManager.
        """
        left_arrays = self.arrays
        right_arrays = other.arrays
        result_arrays = [array_op(left, right) for left, right in zip(left_arrays, right_arrays)]
        return type(self)(result_arrays, self._axes)

    def quantile(self, *, qs: Index, transposed: bool=False, interpolation: QuantileInterpolation='linear') -> ArrayManager:
        arrs = [ensure_block_shape(x, 2) for x in self.arrays]
        new_arrs = [quantile_compat(x, np.asarray(qs._values), interpolation) for x in arrs]
        for i, arr in enumerate(new_arrs):
            if arr.ndim == 2:
                assert arr.shape[0] == 1, arr.shape
                new_arrs[i] = arr[0]
        axes = [qs, self._axes[1]]
        return type(self)(new_arrs, axes)

    def unstack(self, unstacker, fill_value) -> ArrayManager:
        """
        Return a BlockManager with all blocks unstacked.

        Parameters
        ----------
        unstacker : reshape._Unstacker
        fill_value : Any
            fill_value for newly introduced missing values.

        Returns
        -------
        unstacked : BlockManager
        """
        indexer, _ = unstacker._indexer_and_to_sort
        if unstacker.mask.all():
            new_indexer = indexer
            allow_fill = False
            new_mask2D = None
            needs_masking = None
        else:
            new_indexer = np.full(unstacker.mask.shape, -1)
            new_indexer[unstacker.mask] = indexer
            allow_fill = True
            new_mask2D = (~unstacker.mask).reshape(*unstacker.full_shape)
            needs_masking = new_mask2D.any(axis=0)
        new_indexer2D = new_indexer.reshape(*unstacker.full_shape)
        new_indexer2D = ensure_platform_int(new_indexer2D)
        new_arrays = []
        for arr in self.arrays:
            for i in range(unstacker.full_shape[1]):
                if allow_fill:
                    new_arr = take_1d(arr, new_indexer2D[:, i], allow_fill=needs_masking[i], fill_value=fill_value, mask=new_mask2D[:, i])
                else:
                    new_arr = take_1d(arr, new_indexer2D[:, i], allow_fill=False)
                new_arrays.append(new_arr)
        new_index = unstacker.new_index
        new_columns = unstacker.get_new_columns(self._axes[1])
        new_axes = [new_index, new_columns]
        return type(self)(new_arrays, new_axes, verify_integrity=False)

    def as_array(self, dtype=None, copy: bool=False, na_value: object=lib.no_default) -> np.ndarray:
        """
        Convert the blockmanager data into an numpy array.

        Parameters
        ----------
        dtype : object, default None
            Data type of the return array.
        copy : bool, default False
            If True then guarantee that a copy is returned. A value of
            False does not guarantee that the underlying data is not
            copied.
        na_value : object, default lib.no_default
            Value to be used as the missing value sentinel.

        Returns
        -------
        arr : ndarray
        """
        if len(self.arrays) == 0:
            empty_arr = np.empty(self.shape, dtype=float)
            return empty_arr.transpose()
        copy = copy or na_value is not lib.no_default
        if not dtype:
            dtype = interleaved_dtype([arr.dtype for arr in self.arrays])
        dtype = ensure_np_dtype(dtype)
        result = np.empty(self.shape_proper, dtype=dtype)
        for i, arr in enumerate(self.arrays):
            arr = arr.astype(dtype, copy=copy)
            result[:, i] = arr
        if na_value is not lib.no_default:
            result[isna(result)] = na_value
        return result

    @classmethod
    def concat_horizontal(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        """
        Concatenate uniformly-indexed ArrayManagers horizontally.
        """
        arrays = list(itertools.chain.from_iterable([mgr.arrays for mgr in mgrs]))
        new_mgr = cls(arrays, [axes[1], axes[0]], verify_integrity=False)
        return new_mgr

    @classmethod
    def concat_vertical(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        """
        Concatenate uniformly-indexed ArrayManagers vertically.
        """
        arrays = [concat_arrays([mgrs[i].arrays[j] for i in range(len(mgrs))]) for j in range(len(mgrs[0].arrays))]
        new_mgr = cls(arrays, [axes[1], axes[0]], verify_integrity=False)
        return new_mgr