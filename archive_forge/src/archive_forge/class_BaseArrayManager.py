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
class BaseArrayManager(DataManager):
    """
    Core internal data structure to implement DataFrame and Series.

    Alternative to the BlockManager, storing a list of 1D arrays instead of
    Blocks.

    This is *not* a public API class

    Parameters
    ----------
    arrays : Sequence of arrays
    axes : Sequence of Index
    verify_integrity : bool, default True

    """
    __slots__ = ['_axes', 'arrays']
    arrays: list[np.ndarray | ExtensionArray]
    _axes: list[Index]

    def __init__(self, arrays: list[np.ndarray | ExtensionArray], axes: list[Index], verify_integrity: bool=True) -> None:
        raise NotImplementedError

    def make_empty(self, axes=None) -> Self:
        """Return an empty ArrayManager with the items axis of len 0 (no columns)"""
        if axes is None:
            axes = [self.axes[1:], Index([])]
        arrays: list[np.ndarray | ExtensionArray] = []
        return type(self)(arrays, axes)

    @property
    def items(self) -> Index:
        return self._axes[-1]

    @property
    def axes(self) -> list[Index]:
        """Axes is BlockManager-compatible order (columns, rows)"""
        return [self._axes[1], self._axes[0]]

    @property
    def shape_proper(self) -> tuple[int, ...]:
        return tuple((len(ax) for ax in self._axes))

    @staticmethod
    def _normalize_axis(axis: AxisInt) -> int:
        axis = 1 if axis == 0 else 0
        return axis

    def set_axis(self, axis: AxisInt, new_labels: Index) -> None:
        self._validate_set_axis(axis, new_labels)
        axis = self._normalize_axis(axis)
        self._axes[axis] = new_labels

    def get_dtypes(self) -> npt.NDArray[np.object_]:
        return np.array([arr.dtype for arr in self.arrays], dtype='object')

    def add_references(self, mgr: BaseArrayManager) -> None:
        """
        Only implemented on the BlockManager level
        """
        return

    def __getstate__(self):
        return (self.arrays, self._axes)

    def __setstate__(self, state) -> None:
        self.arrays = state[0]
        self._axes = state[1]

    def __repr__(self) -> str:
        output = type(self).__name__
        output += f'\nIndex: {self._axes[0]}'
        if self.ndim == 2:
            output += f'\nColumns: {self._axes[1]}'
        output += f'\n{len(self.arrays)} arrays:'
        for arr in self.arrays:
            output += f'\n{arr.dtype}'
        return output

    def apply(self, f, align_keys: list[str] | None=None, **kwargs) -> Self:
        """
        Iterate over the arrays, collect and create a new ArrayManager.

        Parameters
        ----------
        f : str or callable
            Name of the Array method to apply.
        align_keys: List[str] or None, default None
        **kwargs
            Keywords to pass to `f`

        Returns
        -------
        ArrayManager
        """
        assert 'filter' not in kwargs
        align_keys = align_keys or []
        result_arrays: list[ArrayLike] = []
        aligned_args = {k: kwargs[k] for k in align_keys}
        if f == 'apply':
            f = kwargs.pop('func')
        for i, arr in enumerate(self.arrays):
            if aligned_args:
                for k, obj in aligned_args.items():
                    if isinstance(obj, (ABCSeries, ABCDataFrame)):
                        if obj.ndim == 1:
                            kwargs[k] = obj.iloc[i]
                        else:
                            kwargs[k] = obj.iloc[:, i]._values
                    else:
                        kwargs[k] = obj[i]
            if callable(f):
                applied = f(arr, **kwargs)
            else:
                applied = getattr(arr, f)(**kwargs)
            result_arrays.append(applied)
        new_axes = self._axes
        return type(self)(result_arrays, new_axes)

    def apply_with_block(self, f, align_keys=None, **kwargs) -> Self:
        swap_axis = True
        if f == 'interpolate':
            swap_axis = False
        if swap_axis and 'axis' in kwargs and (self.ndim == 2):
            kwargs['axis'] = 1 if kwargs['axis'] == 0 else 0
        align_keys = align_keys or []
        aligned_args = {k: kwargs[k] for k in align_keys}
        result_arrays = []
        for i, arr in enumerate(self.arrays):
            if aligned_args:
                for k, obj in aligned_args.items():
                    if isinstance(obj, (ABCSeries, ABCDataFrame)):
                        if obj.ndim == 1:
                            if self.ndim == 2:
                                kwargs[k] = obj.iloc[slice(i, i + 1)]._values
                            else:
                                kwargs[k] = obj.iloc[:]._values
                        else:
                            kwargs[k] = obj.iloc[:, [i]]._values
                    elif obj.ndim == 2:
                        kwargs[k] = obj[[i]]
            if isinstance(arr.dtype, np.dtype) and (not isinstance(arr, np.ndarray)):
                arr = np.asarray(arr)
            arr = maybe_coerce_values(arr)
            if self.ndim == 2:
                arr = ensure_block_shape(arr, 2)
                bp = BlockPlacement(slice(0, 1, 1))
                block = new_block(arr, placement=bp, ndim=2)
            else:
                bp = BlockPlacement(slice(0, len(self), 1))
                block = new_block(arr, placement=bp, ndim=1)
            applied = getattr(block, f)(**kwargs)
            if isinstance(applied, list):
                applied = applied[0]
            arr = applied.values
            if self.ndim == 2 and arr.ndim == 2:
                assert len(arr) == 1
                arr = arr[0, :]
            result_arrays.append(arr)
        return type(self)(result_arrays, self._axes)

    def setitem(self, indexer, value, warn: bool=True) -> Self:
        return self.apply_with_block('setitem', indexer=indexer, value=value)

    def diff(self, n: int) -> Self:
        assert self.ndim == 2
        return self.apply(algos.diff, n=n)

    def astype(self, dtype, copy: bool | None=False, errors: str='raise') -> Self:
        if copy is None:
            copy = True
        return self.apply(astype_array_safe, dtype=dtype, copy=copy, errors=errors)

    def convert(self, copy: bool | None) -> Self:
        if copy is None:
            copy = True

        def _convert(arr):
            if is_object_dtype(arr.dtype):
                arr = np.asarray(arr)
                result = lib.maybe_convert_objects(arr, convert_non_numeric=True)
                if result is arr and copy:
                    return arr.copy()
                return result
            else:
                return arr.copy() if copy else arr
        return self.apply(_convert)

    def get_values_for_csv(self, *, float_format, date_format, decimal, na_rep: str='nan', quoting=None) -> Self:
        return self.apply(get_values_for_csv, na_rep=na_rep, quoting=quoting, float_format=float_format, date_format=date_format, decimal=decimal)

    @property
    def any_extension_types(self) -> bool:
        """Whether any of the blocks in this manager are extension blocks"""
        return False

    @property
    def is_view(self) -> bool:
        """return a boolean if we are a single block and are a view"""
        return False

    @property
    def is_single_block(self) -> bool:
        return len(self.arrays) == 1

    def _get_data_subset(self, predicate: Callable) -> Self:
        indices = [i for i, arr in enumerate(self.arrays) if predicate(arr)]
        arrays = [self.arrays[i] for i in indices]
        taker = np.array(indices, dtype='intp')
        new_cols = self._axes[1].take(taker)
        new_axes = [self._axes[0], new_cols]
        return type(self)(arrays, new_axes, verify_integrity=False)

    def get_bool_data(self, copy: bool=False) -> Self:
        """
        Select columns that are bool-dtype and object-dtype columns that are all-bool.

        Parameters
        ----------
        copy : bool, default False
            Whether to copy the blocks
        """
        return self._get_data_subset(lambda x: x.dtype == np.dtype(bool))

    def get_numeric_data(self, copy: bool=False) -> Self:
        """
        Select columns that have a numeric dtype.

        Parameters
        ----------
        copy : bool, default False
            Whether to copy the blocks
        """
        return self._get_data_subset(lambda arr: is_numeric_dtype(arr.dtype) or getattr(arr.dtype, '_is_numeric', False))

    def copy(self, deep: bool | Literal['all'] | None=True) -> Self:
        """
        Make deep or shallow copy of ArrayManager

        Parameters
        ----------
        deep : bool or string, default True
            If False, return shallow copy (do not copy data)
            If 'all', copy data and a deep copy of the index

        Returns
        -------
        BlockManager
        """
        if deep is None:
            deep = True
        if deep:

            def copy_func(ax):
                return ax.copy(deep=True) if deep == 'all' else ax.view()
            new_axes = [copy_func(ax) for ax in self._axes]
        else:
            new_axes = list(self._axes)
        if deep:
            new_arrays = [arr.copy() for arr in self.arrays]
        else:
            new_arrays = list(self.arrays)
        return type(self)(new_arrays, new_axes, verify_integrity=False)

    def reindex_indexer(self, new_axis, indexer, axis: AxisInt, fill_value=None, allow_dups: bool=False, copy: bool | None=True, only_slice: bool=False, use_na_proxy: bool=False) -> Self:
        axis = self._normalize_axis(axis)
        return self._reindex_indexer(new_axis, indexer, axis, fill_value, allow_dups, copy, use_na_proxy)

    def _reindex_indexer(self, new_axis, indexer: npt.NDArray[np.intp] | None, axis: AxisInt, fill_value=None, allow_dups: bool=False, copy: bool | None=True, use_na_proxy: bool=False) -> Self:
        """
        Parameters
        ----------
        new_axis : Index
        indexer : ndarray[intp] or None
        axis : int
        fill_value : object, default None
        allow_dups : bool, default False
        copy : bool, default True


        pandas-indexer with -1's only.
        """
        if copy is None:
            copy = True
        if indexer is None:
            if new_axis is self._axes[axis] and (not copy):
                return self
            result = self.copy(deep=copy)
            result._axes = list(self._axes)
            result._axes[axis] = new_axis
            return result
        if not allow_dups:
            self._axes[axis]._validate_can_reindex(indexer)
        if axis >= self.ndim:
            raise IndexError('Requested axis not found in manager')
        if axis == 1:
            new_arrays = []
            for i in indexer:
                if i == -1:
                    arr = self._make_na_array(fill_value=fill_value, use_na_proxy=use_na_proxy)
                else:
                    arr = self.arrays[i]
                    if copy:
                        arr = arr.copy()
                new_arrays.append(arr)
        else:
            validate_indices(indexer, len(self._axes[0]))
            indexer = ensure_platform_int(indexer)
            mask = indexer == -1
            needs_masking = mask.any()
            new_arrays = [take_1d(arr, indexer, allow_fill=needs_masking, fill_value=fill_value, mask=mask) for arr in self.arrays]
        new_axes = list(self._axes)
        new_axes[axis] = new_axis
        return type(self)(new_arrays, new_axes, verify_integrity=False)

    def take(self, indexer: npt.NDArray[np.intp], axis: AxisInt=1, verify: bool=True) -> Self:
        """
        Take items along any axis.
        """
        assert isinstance(indexer, np.ndarray), type(indexer)
        assert indexer.dtype == np.intp, indexer.dtype
        axis = self._normalize_axis(axis)
        if not indexer.ndim == 1:
            raise ValueError('indexer should be 1-dimensional')
        n = self.shape_proper[axis]
        indexer = maybe_convert_indices(indexer, n, verify=verify)
        new_labels = self._axes[axis].take(indexer)
        return self._reindex_indexer(new_axis=new_labels, indexer=indexer, axis=axis, allow_dups=True)

    def _make_na_array(self, fill_value=None, use_na_proxy: bool=False):
        if use_na_proxy:
            assert fill_value is None
            return NullArrayProxy(self.shape_proper[0])
        if fill_value is None:
            fill_value = np.nan
        dtype, fill_value = infer_dtype_from_scalar(fill_value)
        array_values = make_na_array(dtype, self.shape_proper[:1], fill_value)
        return array_values

    def _equal_values(self, other) -> bool:
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """
        for left, right in zip(self.arrays, other.arrays):
            if not array_equals(left, right):
                return False
        return True