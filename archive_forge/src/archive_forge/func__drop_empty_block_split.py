import itertools
import logging
from typing import Iterable, List, Tuple, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import (
from ray.types import ObjectRef
def _drop_empty_block_split(block_split_indices: List[int], num_rows: int) -> List[int]:
    """drop split indices that creates empty block split. This could happen when there
    are duplicated indices, or index equal to 0 (start of the block) or num_block_rows
    (end of the block).
    """
    prev_index = -1
    optimized_indices = []
    for index in block_split_indices:
        if index == 0 or index == num_rows:
            continue
        if index == prev_index:
            continue
        optimized_indices.append(index)
        prev_index = index
    return optimized_indices