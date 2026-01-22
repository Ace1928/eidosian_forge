from typing import Optional
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule
def _propagate_target_max_block_size_to_upstream_ops(self, dag: PhysicalOperator, target_max_block_size: Optional[int]=None):
    if dag.target_max_block_size is not None:
        target_max_block_size = dag.target_max_block_size
    elif target_max_block_size is not None:
        dag.set_target_max_block_size(target_max_block_size)
    for upstream_op in dag.input_dependencies:
        self._propagate_target_max_block_size_to_upstream_ops(upstream_op, target_max_block_size)