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
class SingleBlockManager(BaseBlockManager, SingleDataManager):
    """manage a single block with"""

    @property
    def ndim(self) -> Literal[1]:
        return 1
    _is_consolidated = True
    _known_consolidated = True
    __slots__ = ()
    is_single_block = True

    def __init__(self, block: Block, axis: Index, verify_integrity: bool=False) -> None:
        self.axes = [axis]
        self.blocks = (block,)

    @classmethod
    def from_blocks(cls, blocks: list[Block], axes: list[Index]) -> Self:
        """
        Constructor for BlockManager and SingleBlockManager with same signature.
        """
        assert len(blocks) == 1
        assert len(axes) == 1
        return cls(blocks[0], axes[0], verify_integrity=False)

    @classmethod
    def from_array(cls, array: ArrayLike, index: Index, refs: BlockValuesRefs | None=None) -> SingleBlockManager:
        """
        Constructor for if we have an array that is not yet a Block.
        """
        array = maybe_coerce_values(array)
        bp = BlockPlacement(slice(0, len(index)))
        block = new_block(array, placement=bp, ndim=1, refs=refs)
        return cls(block, index)

    def to_2d_mgr(self, columns: Index) -> BlockManager:
        """
        Manager analogue of Series.to_frame
        """
        blk = self.blocks[0]
        arr = ensure_block_shape(blk.values, ndim=2)
        bp = BlockPlacement(0)
        new_blk = type(blk)(arr, placement=bp, ndim=2, refs=blk.refs)
        axes = [columns, self.axes[0]]
        return BlockManager([new_blk], axes=axes, verify_integrity=False)

    def _has_no_reference(self, i: int=0) -> bool:
        """
        Check for column `i` if it has references.
        (whether it references another array or is itself being referenced)
        Returns True if the column has no references.
        """
        return not self.blocks[0].refs.has_reference()

    def __getstate__(self):
        block_values = [b.values for b in self.blocks]
        block_items = [self.items[b.mgr_locs.indexer] for b in self.blocks]
        axes_array = list(self.axes)
        extra_state = {'0.14.1': {'axes': axes_array, 'blocks': [{'values': b.values, 'mgr_locs': b.mgr_locs.indexer} for b in self.blocks]}}
        return (axes_array, block_values, block_items, extra_state)

    def __setstate__(self, state) -> None:

        def unpickle_block(values, mgr_locs, ndim: int) -> Block:
            values = extract_array(values, extract_numpy=True)
            if not isinstance(mgr_locs, BlockPlacement):
                mgr_locs = BlockPlacement(mgr_locs)
            values = maybe_coerce_values(values)
            return new_block(values, placement=mgr_locs, ndim=ndim)
        if isinstance(state, tuple) and len(state) >= 4 and ('0.14.1' in state[3]):
            state = state[3]['0.14.1']
            self.axes = [ensure_index(ax) for ax in state['axes']]
            ndim = len(self.axes)
            self.blocks = tuple((unpickle_block(b['values'], b['mgr_locs'], ndim=ndim) for b in state['blocks']))
        else:
            raise NotImplementedError('pre-0.14.1 pickles are no longer supported')
        self._post_setstate()

    def _post_setstate(self) -> None:
        pass

    @cache_readonly
    def _block(self) -> Block:
        return self.blocks[0]

    @property
    def _blknos(self):
        """compat with BlockManager"""
        return None

    @property
    def _blklocs(self):
        """compat with BlockManager"""
        return None

    def get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> Self:
        blk = self._block
        if using_copy_on_write() and len(indexer) > 0 and indexer.all():
            return type(self)(blk.copy(deep=False), self.index)
        array = blk.values[indexer]
        if isinstance(indexer, np.ndarray) and indexer.dtype.kind == 'b':
            refs = None
        else:
            refs = blk.refs
        bp = BlockPlacement(slice(0, len(array)))
        block = type(blk)(array, placement=bp, ndim=1, refs=refs)
        new_idx = self.index[indexer]
        return type(self)(block, new_idx)

    def get_slice(self, slobj: slice, axis: AxisInt=0) -> SingleBlockManager:
        if axis >= self.ndim:
            raise IndexError('Requested axis not found in manager')
        blk = self._block
        array = blk.values[slobj]
        bp = BlockPlacement(slice(0, len(array)))
        block = type(blk)(array, placement=bp, ndim=1, refs=blk.refs)
        new_index = self.index._getitem_slice(slobj)
        return type(self)(block, new_index)

    @property
    def index(self) -> Index:
        return self.axes[0]

    @property
    def dtype(self) -> DtypeObj:
        return self._block.dtype

    def get_dtypes(self) -> npt.NDArray[np.object_]:
        return np.array([self._block.dtype], dtype=object)

    def external_values(self):
        """The array that Series.values returns"""
        return self._block.external_values()

    def internal_values(self):
        """The array that Series._values returns"""
        return self._block.values

    def array_values(self) -> ExtensionArray:
        """The array that Series.array returns"""
        return self._block.array_values

    def get_numeric_data(self) -> Self:
        if self._block.is_numeric:
            return self.copy(deep=False)
        return self.make_empty()

    @property
    def _can_hold_na(self) -> bool:
        return self._block._can_hold_na

    def setitem_inplace(self, indexer, value, warn: bool=True) -> None:
        """
        Set values with indexer.

        For Single[Block/Array]Manager, this backs s[indexer] = value

        This is an inplace version of `setitem()`, mutating the manager/values
        in place, not returning a new Manager (and Block), and thus never changing
        the dtype.
        """
        using_cow = using_copy_on_write()
        warn_cow = warn_copy_on_write()
        if (using_cow or warn_cow) and (not self._has_no_reference(0)):
            if using_cow:
                self.blocks = (self._block.copy(),)
                self._cache.clear()
            elif warn_cow and warn:
                warnings.warn(COW_WARNING_SETITEM_MSG, FutureWarning, stacklevel=find_stack_level())
        super().setitem_inplace(indexer, value)

    def idelete(self, indexer) -> SingleBlockManager:
        """
        Delete single location from SingleBlockManager.

        Ensures that self.blocks doesn't become empty.
        """
        nb = self._block.delete(indexer)[0]
        self.blocks = (nb,)
        self.axes[0] = self.axes[0].delete(indexer)
        self._cache.clear()
        return self

    def fast_xs(self, loc):
        """
        fast path for getting a cross-section
        return a view of the data
        """
        raise NotImplementedError('Use series._values[loc] instead')

    def set_values(self, values: ArrayLike) -> None:
        """
        Set the values of the single block in place.

        Use at your own risk! This does not check if the passed values are
        valid for the current Block/SingleBlockManager (length, dtype, etc),
        and this does not properly keep track of references.
        """
        self.blocks[0].values = values
        self.blocks[0]._mgr_locs = BlockPlacement(slice(len(values)))

    def _equal_values(self, other: Self) -> bool:
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """
        if other.ndim != 1:
            return False
        left = self.blocks[0].values
        right = other.blocks[0].values
        return array_equals(left, right)