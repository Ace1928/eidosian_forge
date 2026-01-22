from __future__ import annotations
from collections.abc import (
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.tslibs import Timestamp
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import infer_dtype_from_scalar
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import (
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import (
from pandas.core.internals.base import (
from pandas.core.internals.blocks import (
from pandas.core.internals.ops import (
def _slice_take_blocks_ax0(self, slice_or_indexer: slice | np.ndarray, fill_value=lib.no_default, only_slice: bool=False, *, use_na_proxy: bool=False, ref_inplace_op: bool=False) -> list[Block]:
    """
        Slice/take blocks along axis=0.

        Overloaded for SingleBlock

        Parameters
        ----------
        slice_or_indexer : slice or np.ndarray[int64]
        fill_value : scalar, default lib.no_default
        only_slice : bool, default False
            If True, we always return views on existing arrays, never copies.
            This is used when called from ops.blockwise.operate_blockwise.
        use_na_proxy : bool, default False
            Whether to use a np.void ndarray for newly introduced columns.
        ref_inplace_op: bool, default False
            Don't track refs if True because we operate inplace

        Returns
        -------
        new_blocks : list of Block
        """
    allow_fill = fill_value is not lib.no_default
    sl_type, slobj, sllen = _preprocess_slice_or_indexer(slice_or_indexer, self.shape[0], allow_fill=allow_fill)
    if self.is_single_block:
        blk = self.blocks[0]
        if sl_type == 'slice':
            if sllen == 0:
                return []
            bp = BlockPlacement(slice(0, sllen))
            return [blk.getitem_block_columns(slobj, new_mgr_locs=bp)]
        elif not allow_fill or self.ndim == 1:
            if allow_fill and fill_value is None:
                fill_value = blk.fill_value
            if not allow_fill and only_slice:
                blocks = [blk.getitem_block_columns(slice(ml, ml + 1), new_mgr_locs=BlockPlacement(i), ref_inplace_op=ref_inplace_op) for i, ml in enumerate(slobj)]
                return blocks
            else:
                bp = BlockPlacement(slice(0, sllen))
                return [blk.take_nd(slobj, axis=0, new_mgr_locs=bp, fill_value=fill_value)]
    if sl_type == 'slice':
        blknos = self.blknos[slobj]
        blklocs = self.blklocs[slobj]
    else:
        blknos = algos.take_nd(self.blknos, slobj, fill_value=-1, allow_fill=allow_fill)
        blklocs = algos.take_nd(self.blklocs, slobj, fill_value=-1, allow_fill=allow_fill)
    blocks = []
    group = not only_slice
    for blkno, mgr_locs in libinternals.get_blkno_placements(blknos, group=group):
        if blkno == -1:
            blocks.append(self._make_na_block(placement=mgr_locs, fill_value=fill_value, use_na_proxy=use_na_proxy))
        else:
            blk = self.blocks[blkno]
            if not blk._can_consolidate and (not blk._validate_ndim):
                deep = not (only_slice or using_copy_on_write())
                for mgr_loc in mgr_locs:
                    newblk = blk.copy(deep=deep)
                    newblk.mgr_locs = BlockPlacement(slice(mgr_loc, mgr_loc + 1))
                    blocks.append(newblk)
            else:
                taker = blklocs[mgr_locs.indexer]
                max_len = max(len(mgr_locs), taker.max() + 1)
                if only_slice or using_copy_on_write():
                    taker = lib.maybe_indices_to_slice(taker, max_len)
                if isinstance(taker, slice):
                    nb = blk.getitem_block_columns(taker, new_mgr_locs=mgr_locs)
                    blocks.append(nb)
                elif only_slice:
                    for i, ml in zip(taker, mgr_locs):
                        slc = slice(i, i + 1)
                        bp = BlockPlacement(ml)
                        nb = blk.getitem_block_columns(slc, new_mgr_locs=bp)
                        blocks.append(nb)
                else:
                    nb = blk.take_nd(taker, axis=0, new_mgr_locs=mgr_locs)
                    blocks.append(nb)
    return blocks