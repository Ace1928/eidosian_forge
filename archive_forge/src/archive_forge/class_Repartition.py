from typing import Any, Dict, List, Optional
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskSpec
from ray.data._internal.planner.exchange.shuffle_task_spec import ShuffleTaskSpec
from ray.data._internal.planner.exchange.sort_task_spec import SortTaskSpec
from ray.data._internal.sort import SortKey
from ray.data.aggregate import AggregateFn
class Repartition(AbstractAllToAll):
    """Logical operator for repartition."""

    def __init__(self, input_op: LogicalOperator, num_outputs: int, shuffle: bool):
        if shuffle:
            sub_progress_bar_names = [ExchangeTaskSpec.MAP_SUB_PROGRESS_BAR_NAME, ExchangeTaskSpec.REDUCE_SUB_PROGRESS_BAR_NAME]
        else:
            sub_progress_bar_names = [ShuffleTaskSpec.SPLIT_REPARTITION_SUB_PROGRESS_BAR_NAME]
        super().__init__('Repartition', input_op, num_outputs=num_outputs, sub_progress_bar_names=sub_progress_bar_names)
        self._shuffle = shuffle