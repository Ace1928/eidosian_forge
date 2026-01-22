from typing import Any, Dict, List, Optional
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskSpec
from ray.data._internal.planner.exchange.shuffle_task_spec import ShuffleTaskSpec
from ray.data._internal.planner.exchange.sort_task_spec import SortTaskSpec
from ray.data._internal.sort import SortKey
from ray.data.aggregate import AggregateFn
class AbstractAllToAll(LogicalOperator):
    """Abstract class for logical operators should be converted to physical
    AllToAllOperator.
    """

    def __init__(self, name: str, input_op: LogicalOperator, num_outputs: Optional[int]=None, sub_progress_bar_names: Optional[List[str]]=None, ray_remote_args: Optional[Dict[str, Any]]=None):
        """
        Args:
            name: Name for this operator. This is the name that will appear when
                inspecting the logical plan of a Dataset.
            input_op: The operator preceding this operator in the plan DAG. The outputs
                of `input_op` will be the inputs to this operator.
            num_outputs: The number of expected output bundles outputted by this
                operator.
            ray_remote_args: Args to provide to ray.remote.
        """
        super().__init__(name, [input_op])
        self._num_outputs = num_outputs
        self._ray_remote_args = ray_remote_args or {}
        self._sub_progress_bar_names = sub_progress_bar_names