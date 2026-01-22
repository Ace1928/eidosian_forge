from __future__ import annotations
from typing import (
from pandas.core.dtypes.common import is_1d_only_ea_dtype
class BlockPairInfo(NamedTuple):
    lvals: ArrayLike
    rvals: ArrayLike
    locs: BlockPlacement
    left_ea: bool
    right_ea: bool
    rblk: Block