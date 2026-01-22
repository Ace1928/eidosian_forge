import itertools
import logging
from typing import Iterable, List, Tuple, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import (
from ray.types import ObjectRef
def _split_all_blocks(blocks_with_metadata: List[Tuple[ObjectRef[Block], BlockMetadata]], per_block_split_indices: List[List[int]], owned_by_consumer: bool) -> Iterable[Tuple[ObjectRef[Block], BlockMetadata]]:
    """Split all the input blocks based on the split indices"""
    split_single_block = cached_remote_fn(_split_single_block)
    all_blocks_split_results: List[BlockPartition] = [None] * len(blocks_with_metadata)
    per_block_split_metadata_futures = []
    per_block_split_block_refs = []
    blocks_splitted = []
    for block_id, block_split_indices in enumerate(per_block_split_indices):
        block_ref, meta = blocks_with_metadata[block_id]
        block_row = meta.num_rows
        block_split_indices = _drop_empty_block_split(block_split_indices, block_row)
        if len(block_split_indices) == 0:
            all_blocks_split_results[block_id] = [(block_ref, meta)]
        else:
            object_refs = split_single_block.options(scheduling_strategy='SPREAD', num_returns=2 + len(block_split_indices)).remote(block_id, block_ref, meta, block_split_indices)
            per_block_split_metadata_futures.append(object_refs[0])
            per_block_split_block_refs.append(object_refs[1:])
            blocks_splitted.append(block_ref)
    if per_block_split_metadata_futures:
        per_block_split_metadata = ray.get(per_block_split_metadata_futures)
        for (block_id, meta), block_refs in zip(per_block_split_metadata, per_block_split_block_refs):
            assert len(meta) == len(block_refs)
            all_blocks_split_results[block_id] = zip(block_refs, meta)
    if owned_by_consumer:
        for b in blocks_splitted:
            trace_deallocation(b, 'split._split_all_blocks')
    else:
        for b in blocks_splitted:
            trace_deallocation(b, 'split._split_all_blocks', free=False)
    return itertools.chain.from_iterable(all_blocks_split_results)