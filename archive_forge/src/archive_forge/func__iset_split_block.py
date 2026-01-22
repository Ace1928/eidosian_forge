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
def _iset_split_block(self, blkno_l: int, blk_locs: np.ndarray | list[int], value: ArrayLike | None=None, refs: BlockValuesRefs | None=None) -> None:
    """Removes columns from a block by splitting the block.

        Avoids copying the whole block through slicing and updates the manager
        after determinint the new block structure. Optionally adds a new block,
        otherwise has to be done by the caller.

        Parameters
        ----------
        blkno_l: The block number to operate on, relevant for updating the manager
        blk_locs: The locations of our block that should be deleted.
        value: The value to set as a replacement.
        refs: The reference tracking object of the value to set.
        """
    blk = self.blocks[blkno_l]
    if self._blklocs is None:
        self._rebuild_blknos_and_blklocs()
    nbs_tup = tuple(blk.delete(blk_locs))
    if value is not None:
        locs = blk.mgr_locs.as_array[blk_locs]
        first_nb = new_block_2d(value, BlockPlacement(locs), refs=refs)
    else:
        first_nb = nbs_tup[0]
        nbs_tup = tuple(nbs_tup[1:])
    nr_blocks = len(self.blocks)
    blocks_tup = self.blocks[:blkno_l] + (first_nb,) + self.blocks[blkno_l + 1:] + nbs_tup
    self.blocks = blocks_tup
    if not nbs_tup and value is not None:
        return
    self._blklocs[first_nb.mgr_locs.indexer] = np.arange(len(first_nb))
    for i, nb in enumerate(nbs_tup):
        self._blklocs[nb.mgr_locs.indexer] = np.arange(len(nb))
        self._blknos[nb.mgr_locs.indexer] = i + nr_blocks