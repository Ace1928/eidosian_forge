from typing import Any, Iterator, Tuple
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import ActorPoolStrategy, get_compute
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.limit_operator import LimitOperator
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.execution.util import make_callable_class_concurrent
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.interfaces.logical_plan import LogicalPlan
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.optimizers import get_execution_plan
from ray.data._internal.logical.rules.set_read_parallelism import (
from ray.data._internal.logical.util import record_operators_usage
from ray.data._internal.memory_tracing import trace_allocation
from ray.data._internal.plan import AllToAllStage, ExecutionPlan, OneToOneStage, Stage
from ray.data._internal.planner.plan_read_op import (
from ray.data._internal.stage_impl import LimitStage, RandomizeBlocksStage
from ray.data._internal.stats import DatasetStats, StatsDict
from ray.data.block import Block, BlockMetadata, CallableClass, List
from ray.data.context import DataContext
from ray.data.datasource import ReadTask
from ray.types import ObjectRef
def _blocks_to_input_buffer(blocks: BlockList, owns_blocks: bool) -> PhysicalOperator:
    """Translate a block list into an InputBuffer operator.

    Args:
        blocks: The block list to translate.
        owns_blocks: Whether we can take ownership of the input blocks.

    Returns:
        The physical operator representing the input block list.
    """
    if hasattr(blocks, '_tasks'):
        read_tasks = blocks._tasks
        remote_args = blocks._remote_args
        assert all((isinstance(t, ReadTask) for t in read_tasks)), read_tasks
        from ray.data._internal.planner.plan_read_op import cleaned_metadata
        inputs = InputDataBuffer([RefBundle([(ray.put(read_task), cleaned_metadata(read_task))], owns_blocks=False) for read_task in read_tasks])
        for i in inputs._input_data:
            for b in i.blocks:
                trace_allocation(b[0], 'legacy_compat.blocks_to_input_buf[0]')

        def do_read(blocks: Iterator[Block], ctx: TaskContext) -> Iterator[Block]:
            for read_task in blocks:
                yield from read_task()
        task_name = 'Read'
        if isinstance(blocks, LazyBlockList):
            task_name = getattr(blocks, '_read_stage_name', task_name)
        return MapOperator.create(create_map_transformer_from_block_fn(do_read), inputs, name=task_name, target_max_block_size=None, ray_remote_args=remote_args)
    else:
        output = _block_list_to_bundles(blocks, owns_blocks=owns_blocks)
        for i in output:
            for b in i.blocks:
                trace_allocation(b[0], 'legacy_compat.blocks_to_input_buf[1]')
        return InputDataBuffer(output)