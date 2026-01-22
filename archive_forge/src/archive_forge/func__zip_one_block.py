import itertools
from typing import List, Tuple
import ray
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import PhysicalOperator, RefBundle
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.split import _split_at_indices
from ray.data._internal.stats import StatsDict
from ray.data.block import (
def _zip_one_block(block: Block, *other_blocks: Block, inverted: bool=False) -> Tuple[Block, BlockMetadata]:
    """Zip together `block` with `other_blocks`."""
    stats = BlockExecStats.builder()
    builder = DelegatingBlockBuilder()
    for other_block in other_blocks:
        builder.add_block(other_block)
    other_block = builder.build()
    if inverted:
        block, other_block = (other_block, block)
    result = BlockAccessor.for_block(block).zip(other_block)
    br = BlockAccessor.for_block(result)
    return (result, br.get_metadata(input_files=[], exec_stats=stats.build()))