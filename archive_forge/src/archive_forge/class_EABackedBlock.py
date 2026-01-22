from __future__ import annotations
from functools import wraps
import inspect
import re
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import missing
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.replace import (
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays import (
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.computation import expressions
from pandas.core.construction import (
from pandas.core.indexers import check_setitem_lengths
from pandas.core.indexes.base import get_values_for_csv
class EABackedBlock(Block):
    """
    Mixin for Block subclasses backed by ExtensionArray.
    """
    values: ExtensionArray

    @final
    def shift(self, periods: int, fill_value: Any=None) -> list[Block]:
        """
        Shift the block by `periods`.

        Dispatches to underlying ExtensionArray and re-boxes in an
        ExtensionBlock.
        """
        new_values = self.values.T.shift(periods=periods, fill_value=fill_value).T
        return [self.make_block_same_class(new_values)]

    @final
    def setitem(self, indexer, value, using_cow: bool=False):
        """
        Attempt self.values[indexer] = value, possibly creating a new array.

        This differs from Block.setitem by not allowing setitem to change
        the dtype of the Block.

        Parameters
        ----------
        indexer : tuple, list-like, array-like, slice, int
            The subset of self.values to set
        value : object
            The value being set
        using_cow: bool, default False
            Signaling if CoW is used.

        Returns
        -------
        Block

        Notes
        -----
        `indexer` is a direct slice/positional indexer. `value` must
        be a compatible shape.
        """
        orig_indexer = indexer
        orig_value = value
        indexer = self._unwrap_setitem_indexer(indexer)
        value = self._maybe_squeeze_arg(value)
        values = self.values
        if values.ndim == 2:
            values = values.T
        check_setitem_lengths(indexer, value, values)
        try:
            values[indexer] = value
        except (ValueError, TypeError):
            if isinstance(self.dtype, IntervalDtype):
                nb = self.coerce_to_target_dtype(orig_value, warn_on_upcast=True)
                return nb.setitem(orig_indexer, orig_value)
            elif isinstance(self, NDArrayBackedExtensionBlock):
                nb = self.coerce_to_target_dtype(orig_value, warn_on_upcast=True)
                return nb.setitem(orig_indexer, orig_value)
            else:
                raise
        else:
            return self

    @final
    def where(self, other, cond, _downcast: str | bool='infer', using_cow: bool=False) -> list[Block]:
        arr = self.values.T
        cond = extract_bool_array(cond)
        orig_other = other
        orig_cond = cond
        other = self._maybe_squeeze_arg(other)
        cond = self._maybe_squeeze_arg(cond)
        if other is lib.no_default:
            other = self.fill_value
        icond, noop = validate_putmask(arr, ~cond)
        if noop:
            if using_cow:
                return [self.copy(deep=False)]
            return [self.copy()]
        try:
            res_values = arr._where(cond, other).T
        except (ValueError, TypeError):
            if self.ndim == 1 or self.shape[0] == 1:
                if isinstance(self.dtype, IntervalDtype):
                    blk = self.coerce_to_target_dtype(orig_other)
                    nbs = blk.where(orig_other, orig_cond, using_cow=using_cow)
                    return self._maybe_downcast(nbs, downcast=_downcast, using_cow=using_cow, caller='where')
                elif isinstance(self, NDArrayBackedExtensionBlock):
                    blk = self.coerce_to_target_dtype(orig_other)
                    nbs = blk.where(orig_other, orig_cond, using_cow=using_cow)
                    return self._maybe_downcast(nbs, downcast=_downcast, using_cow=using_cow, caller='where')
                else:
                    raise
            else:
                is_array = isinstance(orig_other, (np.ndarray, ExtensionArray))
                res_blocks = []
                nbs = self._split()
                for i, nb in enumerate(nbs):
                    n = orig_other
                    if is_array:
                        n = orig_other[:, i:i + 1]
                    submask = orig_cond[:, i:i + 1]
                    rbs = nb.where(n, submask, using_cow=using_cow)
                    res_blocks.extend(rbs)
                return res_blocks
        nb = self.make_block_same_class(res_values)
        return [nb]

    @final
    def putmask(self, mask, new, using_cow: bool=False, already_warned=None) -> list[Block]:
        """
        See Block.putmask.__doc__
        """
        mask = extract_bool_array(mask)
        if new is lib.no_default:
            new = self.fill_value
        orig_new = new
        orig_mask = mask
        new = self._maybe_squeeze_arg(new)
        mask = self._maybe_squeeze_arg(mask)
        if not mask.any():
            if using_cow:
                return [self.copy(deep=False)]
            return [self]
        if warn_copy_on_write() and already_warned is not None and (not already_warned.warned_already):
            if self.refs.has_reference():
                warnings.warn(COW_WARNING_GENERAL_MSG, FutureWarning, stacklevel=find_stack_level())
                already_warned.warned_already = True
        self = self._maybe_copy(using_cow, inplace=True)
        values = self.values
        if values.ndim == 2:
            values = values.T
        try:
            values._putmask(mask, new)
        except (TypeError, ValueError):
            if self.ndim == 1 or self.shape[0] == 1:
                if isinstance(self.dtype, IntervalDtype):
                    blk = self.coerce_to_target_dtype(orig_new, warn_on_upcast=True)
                    return blk.putmask(orig_mask, orig_new)
                elif isinstance(self, NDArrayBackedExtensionBlock):
                    blk = self.coerce_to_target_dtype(orig_new, warn_on_upcast=True)
                    return blk.putmask(orig_mask, orig_new)
                else:
                    raise
            else:
                is_array = isinstance(orig_new, (np.ndarray, ExtensionArray))
                res_blocks = []
                nbs = self._split()
                for i, nb in enumerate(nbs):
                    n = orig_new
                    if is_array:
                        n = orig_new[:, i:i + 1]
                    submask = orig_mask[:, i:i + 1]
                    rbs = nb.putmask(submask, n)
                    res_blocks.extend(rbs)
                return res_blocks
        return [self]

    @final
    def delete(self, loc) -> list[Block]:
        if self.ndim == 1:
            values = self.values.delete(loc)
            mgr_locs = self._mgr_locs.delete(loc)
            return [type(self)(values, placement=mgr_locs, ndim=self.ndim)]
        elif self.values.ndim == 1:
            return []
        return super().delete(loc)

    @final
    @cache_readonly
    def array_values(self) -> ExtensionArray:
        return self.values

    @final
    def get_values(self, dtype: DtypeObj | None=None) -> np.ndarray:
        """
        return object dtype as boxed values, such as Timestamps/Timedelta
        """
        values: ArrayLike = self.values
        if dtype == _dtype_obj:
            values = values.astype(object)
        return np.asarray(values).reshape(self.shape)

    @final
    def pad_or_backfill(self, *, method: FillnaOptions, axis: AxisInt=0, inplace: bool=False, limit: int | None=None, limit_area: Literal['inside', 'outside'] | None=None, downcast: Literal['infer'] | None=None, using_cow: bool=False, already_warned=None) -> list[Block]:
        values = self.values
        kwargs: dict[str, Any] = {'method': method, 'limit': limit}
        if 'limit_area' in inspect.signature(values._pad_or_backfill).parameters:
            kwargs['limit_area'] = limit_area
        elif limit_area is not None:
            raise NotImplementedError(f'{type(values).__name__} does not implement limit_area (added in pandas 2.2). 3rd-party ExtnsionArray authors need to add this argument to _pad_or_backfill.')
        if values.ndim == 2 and axis == 1:
            new_values = values.T._pad_or_backfill(**kwargs).T
        else:
            new_values = values._pad_or_backfill(**kwargs)
        return [self.make_block_same_class(new_values)]