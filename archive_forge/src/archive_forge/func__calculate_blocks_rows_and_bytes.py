import itertools
from typing import List, Tuple
import ray
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import PhysicalOperator, RefBundle
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.split import _split_at_indices
from ray.data._internal.stats import StatsDict
from ray.data.block import (
def _calculate_blocks_rows_and_bytes(self, blocks_with_metadata: BlockPartition) -> Tuple[List[int], List[int]]:
    """Calculate the number of rows and size in bytes for a list of blocks with
        metadata.
        """
    get_num_rows_and_bytes = cached_remote_fn(_get_num_rows_and_bytes)
    block_rows = []
    block_bytes = []
    for block, metadata in blocks_with_metadata:
        if metadata.num_rows is None or metadata.size_bytes is None:
            num_rows, size_bytes = ray.get(get_num_rows_and_bytes.remote(block))
            metadata.num_rows = num_rows
            metadata.size_bytes = size_bytes
        block_rows.append(metadata.num_rows)
        block_bytes.append(metadata.size_bytes)
    return (block_rows, block_bytes)