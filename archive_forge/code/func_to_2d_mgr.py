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