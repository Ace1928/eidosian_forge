from types import ModuleType
from typing import Any, Callable, Tuple, Union
import numpy as np
from ray.data.block import AggType, Block, KeyType, T, U
def _accum_block_null(a: WrappedAggType, block: Block) -> WrappedAggType:
    ret = accum_block(block)
    if ret is not None:
        ret = _wrap_acc(ret, has_data=True)
    elif ignore_nulls:
        ret = a
    return null_merge(a, ret)