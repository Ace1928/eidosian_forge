import abc
from typing import Optional
from ray.data._internal.logical.interfaces import LogicalOperator
class Limit(AbstractOneToOne):
    """Logical operator for limit."""

    def __init__(self, input_op: LogicalOperator, limit: int):
        super().__init__(f'limit={limit}', input_op)
        self._limit = limit

    @property
    def can_modify_num_rows(self) -> bool:
        return True