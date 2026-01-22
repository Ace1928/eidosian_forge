from typing import List, Optional
from ray.data._internal.execution.interfaces import (
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.stats import StatsDict
class OneToOneOperator(PhysicalOperator):
    """An operator that has one input and one output dependency.

    This operator serves as the base for map, filter, limit, etc.
    """

    def __init__(self, name: str, input_op: PhysicalOperator, target_max_block_size: Optional[int]):
        """Create a OneToOneOperator.
        Args:
            input_op: Operator generating input data for this op.
            name: The name of this operator.
            target_max_block_size: The target maximum number of bytes to
                include in an output block.
        """
        super().__init__(name, [input_op], target_max_block_size)

    @property
    def input_dependency(self) -> PhysicalOperator:
        return self.input_dependencies[0]