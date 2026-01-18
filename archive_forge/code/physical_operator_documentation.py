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
Called periodically by the executor.

        Args:
            input_queue_size: The number of inputs queued outside this operator.
            under_resource_limits: Whether this operator is under resource limits.
        