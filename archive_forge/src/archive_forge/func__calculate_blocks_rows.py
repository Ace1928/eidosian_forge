import itertools
import logging
from typing import Iterable, List, Tuple, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import (
from ray.types import ObjectRef
def _calculate_blocks_rows(blocks_with_metadata: BlockPartition) -> List[int]:
    """Calculate the number of rows for a list of blocks with metadata."""
    get_num_rows = cached_remote_fn(_get_num_rows)
    block_rows = []
    for block, metadata in blocks_with_metadata:
        if metadata.num_rows is None:
            num_rows = ray.get(get_num_rows.remote(block))
            metadata.num_rows = num_rows
        else:
            num_rows = metadata.num_rows
        block_rows.append(num_rows)
    return block_rows