import copy
import functools
import itertools
from typing import (
import ray
from ray._private.internal_api import get_memory_info_reply, get_state_from_address
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import (
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.rules.operator_fusion import _are_remote_args_compatible
from ray.data._internal.logical.rules.set_read_parallelism import (
from ray.data._internal.planner.plan_read_op import (
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary
from ray.data._internal.util import (
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.debug import log_once
def _rewrite_read_stage(in_blocks: LazyBlockList, stages: List[Stage]) -> Tuple[BlockList, DatasetStats, List[Stage]]:
    """Rewrite the read stage to a OneToOne stage over read tasks as input.

    For example, suppose the plan was [Read -> MapBatches(Fn)]. These stages cannot
    be fused, since read stages are handled specially.
    After rewriting to [GetReadTasks -> MapBatches(DoRead) -> MapBatches(Fn)],
    now we can fuse the latter two MapBatches stages into a single OneToOne stage:
    [GetReadTasks -> MapBatches(DoRead -> Fn)].

    Args:
        blocks: Lazy block list representing read stage.
        stages: List of current stages.

    Returns:
        Non-lazy block list containing read tasks for not-yet-read block partitions,
        new stats for the block list, and the new list of stages.
    """
    from ray.data._internal.stage_impl import RandomizeBlocksStage
    remote_args = in_blocks._remote_args
    blocks, metadata = ([], [])
    for read_task in in_blocks._tasks:
        blocks.append(ray.put(read_task._read_fn))
        metadata.append(read_task.get_metadata())
    block_list = BlockList(blocks, metadata, owned_by_consumer=in_blocks._owned_by_consumer)

    @_adapt_for_multiple_blocks
    def block_fn(read_fn: Callable[[], Iterator[Block]], ctx: TaskContext) -> Iterator[Block]:
        for block in read_fn():
            yield block
    name = in_blocks._read_stage_name or 'Read'
    if isinstance(name, list):
        name = '->'.join(name)
    has_randomize = stages and isinstance(stages[0], RandomizeBlocksStage)
    if has_randomize:
        if stages and isinstance(stages[0], RandomizeBlocksStage):
            block_list, _ = stages[0].do_randomize(block_list)
            stages = stages[1:]
        name += '->RandomizeBlockOrder'
    stage = OneToOneStage(name, block_fn, TaskPoolStrategy(), remote_args)
    stats = DatasetStats(stages={}, parent=None)
    stages.insert(0, stage)
    return (block_list, stats, stages)