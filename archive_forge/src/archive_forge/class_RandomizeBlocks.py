from typing import Any, Dict, List, Optional
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskSpec
from ray.data._internal.planner.exchange.shuffle_task_spec import ShuffleTaskSpec
from ray.data._internal.planner.exchange.sort_task_spec import SortTaskSpec
from ray.data._internal.sort import SortKey
from ray.data.aggregate import AggregateFn
class RandomizeBlocks(AbstractAllToAll):
    """Logical operator for randomize_block_order."""

    def __init__(self, input_op: LogicalOperator, seed: Optional[int]=None):
        super().__init__('RandomizeBlockOrder', input_op)
        self._seed = seed