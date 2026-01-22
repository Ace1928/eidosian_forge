import time
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Tuple
import ray
from .backpressure_policy import BackpressurePolicy
from ray.data._internal.dataset_logger import DatasetLogger
def _print_warning(self, op: 'PhysicalOperator', idle_time: float):
    if self._warning_printed:
        return
    self._warning_printed = True
    msg = f'Operator {op} is running but has no outputs for {idle_time} seconds. Execution may be slower than expected.\nIgnore this warning if your UDF is expected to be slow. Otherwise, this can happen when there are fewer cluster resources available to Ray Data than expected. If you have non-Data tasks or actors running in the cluster, exclude their resources from Ray Data with `DataContext.get_current().execution_options.exclude_resources`. This message will only print once.'
    logger.get_logger().warning(msg)