from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
import ray
from .ref_bundle import RefBundle
from ray._raylet import ObjectRefGenerator
from ray.data._internal.execution.interfaces.execution_options import (
from ray.data._internal.execution.interfaces.op_runtime_metrics import OpRuntimeMetrics
from ray.data._internal.logical.interfaces import Operator
from ray.data._internal.stats import StatsDict
from ray.data.context import DataContext
@property
def actual_target_max_block_size(self) -> int:
    """
        The actual target max block size output by this operator.
        """
    target_max_block_size = self._target_max_block_size
    if target_max_block_size is None:
        target_max_block_size = DataContext.get_current().target_max_block_size
    return target_max_block_size