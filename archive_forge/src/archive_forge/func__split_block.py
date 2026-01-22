import math
from collections import deque
from typing import Any, Dict, List, Optional
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.util import locality_string
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.types import ObjectRef
def _split_block(b: ObjectRef[Block], left_size: int) -> (ObjectRef[Block], ObjectRef[Block]):
    split_single_block = cached_remote_fn(_split_single_block)
    left, right = split_single_block.options(num_returns=2).remote(b, left_size)
    return (left, right)