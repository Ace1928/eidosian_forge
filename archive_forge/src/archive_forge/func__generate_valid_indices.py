import itertools
import logging
from typing import Iterable, List, Tuple, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import (
from ray.types import ObjectRef
def _generate_valid_indices(num_rows_per_block: List[int], split_indices: List[int]) -> List[int]:
    """Generate valid split indices by apply min(index, total_num_rows)
    to every index."""
    total_rows = sum(num_rows_per_block)
    return [min(index, total_rows) for index in split_indices]