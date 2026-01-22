import os
import threading
import time
import uuid
from typing import Dict, Iterator, List, Optional
import ray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.autoscaling_requester import (
from ray.data._internal.execution.backpressure_policy import (
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.streaming_executor_state import (
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.stats import DatasetStats, StatsManager
from ray.data.context import DataContext
class StreamIterator(OutputIterator):

    def __init__(self, outer: Executor):
        self._outer = outer

    def get_next(self, output_split_idx: Optional[int]=None) -> RefBundle:
        try:
            item = self._outer._output_node.get_output_blocking(output_split_idx)
            if self._outer._global_info:
                self._outer._global_info.update(1, dag._estimated_output_blocks)
            return item
        except BaseException as e:
            self._outer.shutdown(isinstance(e, StopIteration))
            raise

    def __del__(self):
        self._outer.shutdown()