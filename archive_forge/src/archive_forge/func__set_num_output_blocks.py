from typing import Callable, List, Optional
from ray.data._internal.execution.interfaces import (
from ray.data._internal.stats import StatsDict
def _set_num_output_blocks(self, num_output_blocks):
    self._num_output_blocks = num_output_blocks