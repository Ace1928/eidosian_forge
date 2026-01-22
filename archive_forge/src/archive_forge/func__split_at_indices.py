import itertools
import logging
from typing import Iterable, List, Tuple, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import (
from ray.types import ObjectRef
def _split_at_indices(blocks_with_metadata: List[Tuple[ObjectRef[Block], BlockMetadata]], indices: List[int], owned_by_consumer: bool=True, block_rows: List[int]=None) -> Tuple[List[List[ObjectRef[Block]]], List[List[BlockMetadata]]]:
    """Split blocks at the provided indices.

    Args:
        blocks_with_metadata: Block futures to split, including the associated metadata.
        indices: The (global) indices at which to split the blocks.
        owned_by_consumer: Whether the provided blocks are owned by the consumer.
        block_rows: The number of rows for each block, in case it has already been
            computed.

    Returns:
        The block split futures and their metadata. If an index split is empty, the
        corresponding block split will be empty .
    """
    blocks_with_metadata = list(blocks_with_metadata)
    if len(blocks_with_metadata) == 0:
        return ([[]] * (len(indices) + 1), [[]] * (len(indices) + 1))
    if block_rows is None:
        block_rows = _calculate_blocks_rows(blocks_with_metadata)
    valid_indices = _generate_valid_indices(block_rows, indices)
    per_block_split_indices: List[List[int]] = _generate_per_block_split_indices(block_rows, valid_indices)
    all_blocks_split_results: Iterable[Tuple[ObjectRef[Block], BlockMetadata]] = _split_all_blocks(blocks_with_metadata, per_block_split_indices, owned_by_consumer)
    helper = [0] + valid_indices + [sum(block_rows)]
    split_sizes = [helper[i] - helper[i - 1] for i in range(1, len(helper))]
    return _generate_global_split_results(all_blocks_split_results, split_sizes)