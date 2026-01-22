from typing import List, Optional, Tuple
from ray.data._internal.compute import get_compute, is_task_compute
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.actor_pool_map_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.task_pool_map_operator import (
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule
from ray.data._internal.logical.operators.all_to_all_operator import (
from ray.data._internal.logical.operators.map_operator import AbstractUDFMap
from ray.data._internal.stats import StatsDict
from ray.data.context import DataContext
def _get_fused_all_to_all_operator(self, down_op: AllToAllOperator, up_op: MapOperator) -> AllToAllOperator:
    assert self._can_fuse(down_op, up_op), f'Current rule supports fusing MapOperator -> AllToAllOperator, but received: {type(up_op).__name__} -> {type(down_op).__name__}'
    name = up_op.name + '->' + down_op.name
    down_logical_op: AbstractAllToAll = self._op_map.pop(down_op)
    up_logical_op: AbstractUDFMap = self._op_map.pop(up_op)
    ray_remote_args = up_logical_op._ray_remote_args
    down_transform_fn = down_op.get_transformation_fn()
    up_map_transformer = up_op.get_map_transformer()

    def fused_all_to_all_transform_fn(blocks: List[RefBundle], ctx: TaskContext) -> Tuple[List[RefBundle], StatsDict]:
        """To fuse MapOperator->AllToAllOperator, we store the map function
            in the TaskContext so that it may be used by the downstream
            AllToAllOperator's transform function."""
        ctx.upstream_map_transformer = up_map_transformer
        ctx.upstream_map_ray_remote_args = ray_remote_args
        return down_transform_fn(blocks, ctx)
    input_deps = up_op.input_dependencies
    assert len(input_deps) == 1
    input_op = input_deps[0]
    target_max_block_size = self._get_merged_target_max_block_size(up_op.target_max_block_size, down_op.target_max_block_size)
    op = AllToAllOperator(fused_all_to_all_transform_fn, input_op, target_max_block_size=target_max_block_size, num_outputs=down_op._num_outputs, sub_progress_bar_names=down_op._sub_progress_bar_names, name=name)
    input_op = up_logical_op
    if isinstance(down_logical_op, RandomShuffle):
        logical_op = RandomShuffle(input_op, name=name, ray_remote_args=ray_remote_args)
    elif isinstance(down_logical_op, Repartition):
        logical_op = Repartition(input_op, num_outputs=down_logical_op._num_outputs, shuffle=down_logical_op._shuffle)
    self._op_map[op] = logical_op
    return op