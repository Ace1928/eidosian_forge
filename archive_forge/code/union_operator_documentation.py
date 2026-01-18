from typing import List
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.base_physical_operator import NAryOperator
from ray.data._internal.stats import StatsDict
When `self._preserve_order` is True, change the
        output buffer source to the next input dependency
        once the current input dependency calls `input_done()`.