import math
from collections import deque
from typing import Any, Dict, List, Optional
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.util import locality_string
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.types import ObjectRef
def _can_safely_dispatch(self, target_index: int, nrow: int) -> bool:
    if not self._equal:
        return True
    output_distribution = self._num_output.copy()
    output_distribution[target_index] += nrow
    buffer_requirement = self._calculate_buffer_requirement(output_distribution)
    buffer_size = sum((b.num_rows() for b in self._buffer))
    return buffer_size >= buffer_requirement