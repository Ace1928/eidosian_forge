from typing import List
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.base_physical_operator import NAryOperator
from ray.data._internal.stats import StatsDict
def _get_next_inner(self) -> RefBundle:
    return self._output_buffer.pop(0)