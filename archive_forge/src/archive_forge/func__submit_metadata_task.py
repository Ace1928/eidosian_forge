import copy
import functools
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Set, Union
import ray
from ray import ObjectRef
from ray._raylet import ObjectRefGenerator
from ray.data._internal.compute import (
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.interfaces.physical_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def _submit_metadata_task(self, result_ref: ObjectRef, task_done_callback: Callable[[], None]):
    """Submit a new metadata-handling task."""
    task_index = self._next_metadata_task_idx
    self._next_metadata_task_idx += 1

    def _task_done_callback():
        self._metadata_tasks.pop(task_index)
        task_done_callback()
    self._metadata_tasks[task_index] = MetadataOpTask(task_index, result_ref, _task_done_callback)