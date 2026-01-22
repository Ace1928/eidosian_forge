from __future__ import annotations
from contextlib import suppress
import sys
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs.indexing import NDFrameIndexerBase
from pandas._libs.lib import item_from_zerodim
from pandas.compat import PYPY
from pandas.errors import (
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import algorithms as algos
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.indexes.api import (
class _LocationIndexer(NDFrameIndexerBase):
    _valid_types: str
    axis: AxisInt | None = None
    _takeable: bool

    @final
    def __call__(self, axis: Axis | None=None) -> Self:
        new_self = type(self)(self.name, self.obj)
        if axis is not None:
            axis_int_none = self.obj._get_axis_number(axis)
        else:
            axis_int_none = axis
        new_self.axis = axis_int_none
        return new_self

    def _get_setitem_indexer(self, key):
        """
        Convert a potentially-label-based key into a positional indexer.
        """
        if self.name == 'loc':
            self._ensure_listlike_indexer(key)
        if isinstance(key, tuple):
            for x in key:
                check_dict_or_set_indexers(x)
        if self.axis is not None:
            key = _tupleize_axis_indexer(self.ndim, self.axis, key)
        ax = self.obj._get_axis(0)
        if isinstance(ax, MultiIndex) and self.name != 'iloc' and is_hashable(key) and (not isinstance(key, slice)):
            with suppress(KeyError, InvalidIndexError):
                return ax.get_loc(key)
        if isinstance(key, tuple):
            with suppress(IndexingError):
                return self._convert_tuple(key)
        if isinstance(key, range):
            key = list(key)
        return self._convert_to_indexer(key, axis=0)

    @final
    def _maybe_mask_setitem_value(self, indexer, value):
        """
        If we have obj.iloc[mask] = series_or_frame and series_or_frame has the
        same length as obj, we treat this as obj.iloc[mask] = series_or_frame[mask],
        similar to Series.__setitem__.

        Note this is only for loc, not iloc.
        """
        if isinstance(indexer, tuple) and len(indexer) == 2 and isinstance(value, (ABCSeries, ABCDataFrame)):
            pi, icols = indexer
            ndim = value.ndim
            if com.is_bool_indexer(pi) and len(value) == len(pi):
                newkey = pi.nonzero()[0]
                if is_scalar_indexer(icols, self.ndim - 1) and ndim == 1:
                    value = self.obj.iloc._align_series(indexer, value)
                    indexer = (newkey, icols)
                elif isinstance(icols, np.ndarray) and icols.dtype.kind == 'i' and (len(icols) == 1):
                    if ndim == 1:
                        value = self.obj.iloc._align_series(indexer, value)
                        indexer = (newkey, icols)
                    elif ndim == 2 and value.shape[1] == 1:
                        value = self.obj.iloc._align_frame(indexer, value)
                        indexer = (newkey, icols)
        elif com.is_bool_indexer(indexer):
            indexer = indexer.nonzero()[0]
        return (indexer, value)

    @final
    def _ensure_listlike_indexer(self, key, axis=None, value=None) -> None:
        """
        Ensure that a list-like of column labels are all present by adding them if
        they do not already exist.

        Parameters
        ----------
        key : list-like of column labels
            Target labels.
        axis : key axis if known
        """
        column_axis = 1
        if self.ndim != 2:
            return
        if isinstance(key, tuple) and len(key) > 1:
            key = key[column_axis]
            axis = column_axis
        if axis == column_axis and (not isinstance(self.obj.columns, MultiIndex)) and is_list_like_indexer(key) and (not com.is_bool_indexer(key)) and all((is_hashable(k) for k in key)):
            keys = self.obj.columns.union(key, sort=False)
            diff = Index(key).difference(self.obj.columns, sort=False)
            if len(diff):
                indexer = np.arange(len(keys), dtype=np.intp)
                indexer[len(self.obj.columns):] = -1
                new_mgr = self.obj._mgr.reindex_indexer(keys, indexer=indexer, axis=0, only_slice=True, use_na_proxy=True)
                self.obj._mgr = new_mgr
                return
            self.obj._mgr = self.obj._mgr.reindex_axis(keys, axis=0, only_slice=True)

    @final
    def __setitem__(self, key, value) -> None:
        if not PYPY and using_copy_on_write():
            if sys.getrefcount(self.obj) <= 2:
                warnings.warn(_chained_assignment_msg, ChainedAssignmentError, stacklevel=2)
        elif not PYPY and (not using_copy_on_write()):
            ctr = sys.getrefcount(self.obj)
            ref_count = 2
            if not warn_copy_on_write() and _check_cacher(self.obj):
                ref_count += 1
            if ctr <= ref_count:
                warnings.warn(_chained_assignment_warning_msg, FutureWarning, stacklevel=2)
        check_dict_or_set_indexers(key)
        if isinstance(key, tuple):
            key = tuple((list(x) if is_iterator(x) else x for x in key))
            key = tuple((com.apply_if_callable(x, self.obj) for x in key))
        else:
            maybe_callable = com.apply_if_callable(key, self.obj)
            key = self._check_deprecated_callable_usage(key, maybe_callable)
        indexer = self._get_setitem_indexer(key)
        self._has_valid_setitem_indexer(key)
        iloc = self if self.name == 'iloc' else self.obj.iloc
        iloc._setitem_with_indexer(indexer, value, self.name)

    def _validate_key(self, key, axis: AxisInt):
        """
        Ensure that key is valid for current indexer.

        Parameters
        ----------
        key : scalar, slice or list-like
            Key requested.
        axis : int
            Dimension on which the indexing is being made.

        Raises
        ------
        TypeError
            If the key (or some element of it) has wrong type.
        IndexError
            If the key (or some element of it) is out of bounds.
        KeyError
            If the key was not found.
        """
        raise AbstractMethodError(self)

    @final
    def _expand_ellipsis(self, tup: tuple) -> tuple:
        """
        If a tuple key includes an Ellipsis, replace it with an appropriate
        number of null slices.
        """
        if any((x is Ellipsis for x in tup)):
            if tup.count(Ellipsis) > 1:
                raise IndexingError(_one_ellipsis_message)
            if len(tup) == self.ndim:
                i = tup.index(Ellipsis)
                new_key = tup[:i] + (_NS,) + tup[i + 1:]
                return new_key
        return tup

    @final
    def _validate_tuple_indexer(self, key: tuple) -> tuple:
        """
        Check the key for valid keys across my indexer.
        """
        key = self._validate_key_length(key)
        key = self._expand_ellipsis(key)
        for i, k in enumerate(key):
            try:
                self._validate_key(k, i)
            except ValueError as err:
                raise ValueError(f'Location based indexing can only have [{self._valid_types}] types') from err
        return key

    @final
    def _is_nested_tuple_indexer(self, tup: tuple) -> bool:
        """
        Returns
        -------
        bool
        """
        if any((isinstance(ax, MultiIndex) for ax in self.obj.axes)):
            return any((is_nested_tuple(tup, ax) for ax in self.obj.axes))
        return False

    @final
    def _convert_tuple(self, key: tuple) -> tuple:
        self._validate_key_length(key)
        keyidx = [self._convert_to_indexer(k, axis=i) for i, k in enumerate(key)]
        return tuple(keyidx)

    @final
    def _validate_key_length(self, key: tuple) -> tuple:
        if len(key) > self.ndim:
            if key[0] is Ellipsis:
                key = key[1:]
                if Ellipsis in key:
                    raise IndexingError(_one_ellipsis_message)
                return self._validate_key_length(key)
            raise IndexingError('Too many indexers')
        return key

    @final
    def _getitem_tuple_same_dim(self, tup: tuple):
        """
        Index with indexers that should return an object of the same dimension
        as self.obj.

        This is only called after a failed call to _getitem_lowerdim.
        """
        retval = self.obj
        start_val = self.ndim - len(tup) + 1
        for i, key in enumerate(reversed(tup)):
            i = self.ndim - i - start_val
            if com.is_null_slice(key):
                continue
            retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
            assert retval.ndim == self.ndim
        if retval is self.obj:
            retval = retval.copy(deep=False)
        return retval

    @final
    def _getitem_lowerdim(self, tup: tuple):
        if self.axis is not None:
            axis = self.obj._get_axis_number(self.axis)
            return self._getitem_axis(tup, axis=axis)
        if self._is_nested_tuple_indexer(tup):
            return self._getitem_nested_tuple(tup)
        ax0 = self.obj._get_axis(0)
        if isinstance(ax0, MultiIndex) and self.name != 'iloc' and (not any((isinstance(x, slice) for x in tup))):
            with suppress(IndexingError):
                return cast(_LocIndexer, self)._handle_lowerdim_multi_index_axis0(tup)
        tup = self._validate_key_length(tup)
        for i, key in enumerate(tup):
            if is_label_like(key):
                section = self._getitem_axis(key, axis=i)
                if section.ndim == self.ndim:
                    new_key = tup[:i] + (_NS,) + tup[i + 1:]
                else:
                    new_key = tup[:i] + tup[i + 1:]
                    if len(new_key) == 1:
                        new_key = new_key[0]
                if com.is_null_slice(new_key):
                    return section
                return getattr(section, self.name)[new_key]
        raise IndexingError('not applicable')

    @final
    def _getitem_nested_tuple(self, tup: tuple):

        def _contains_slice(x: object) -> bool:
            if isinstance(x, tuple):
                return any((isinstance(v, slice) for v in x))
            elif isinstance(x, slice):
                return True
            return False
        for key in tup:
            check_dict_or_set_indexers(key)
        if len(tup) > self.ndim:
            if self.name != 'loc':
                raise ValueError('Too many indices')
            if all((is_hashable(x) and (not _contains_slice(x)) or com.is_null_slice(x) for x in tup)):
                with suppress(IndexingError):
                    return cast(_LocIndexer, self)._handle_lowerdim_multi_index_axis0(tup)
            elif isinstance(self.obj, ABCSeries) and any((isinstance(k, tuple) for k in tup)):
                raise IndexingError('Too many indexers')
            axis = self.axis or 0
            return self._getitem_axis(tup, axis=axis)
        obj = self.obj
        axis = len(tup) - 1
        for key in tup[::-1]:
            if com.is_null_slice(key):
                axis -= 1
                continue
            obj = getattr(obj, self.name)._getitem_axis(key, axis=axis)
            axis -= 1
            if is_scalar(obj) or not hasattr(obj, 'ndim'):
                break
        return obj

    def _convert_to_indexer(self, key, axis: AxisInt):
        raise AbstractMethodError(self)

    def _check_deprecated_callable_usage(self, key: Any, maybe_callable: T) -> T:
        if self.name == 'iloc' and callable(key) and isinstance(maybe_callable, tuple):
            warnings.warn('Returning a tuple from a callable with iloc is deprecated and will be removed in a future version', FutureWarning, stacklevel=find_stack_level())
        return maybe_callable

    @final
    def __getitem__(self, key):
        check_dict_or_set_indexers(key)
        if type(key) is tuple:
            key = tuple((list(x) if is_iterator(x) else x for x in key))
            key = tuple((com.apply_if_callable(x, self.obj) for x in key))
            if self._is_scalar_access(key):
                return self.obj._get_value(*key, takeable=self._takeable)
            return self._getitem_tuple(key)
        else:
            axis = self.axis or 0
            maybe_callable = com.apply_if_callable(key, self.obj)
            maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
            return self._getitem_axis(maybe_callable, axis=axis)

    def _is_scalar_access(self, key: tuple):
        raise NotImplementedError()

    def _getitem_tuple(self, tup: tuple):
        raise AbstractMethodError(self)

    def _getitem_axis(self, key, axis: AxisInt):
        raise NotImplementedError()

    def _has_valid_setitem_indexer(self, indexer) -> bool:
        raise AbstractMethodError(self)

    @final
    def _getbool_axis(self, key, axis: AxisInt):
        labels = self.obj._get_axis(axis)
        key = check_bool_indexer(labels, key)
        inds = key.nonzero()[0]
        return self.obj._take_with_is_copy(inds, axis=axis)