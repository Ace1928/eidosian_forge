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
class _UnorderedOutputQueue(_OutputQueue):
    """An queue that does not guarantee output order of finished tasks."""

    def __init__(self):
        self._queue: Deque[RefBundle] = deque()

    def notify_task_output_ready(self, _: int, output: RefBundle):
        self._queue.append(output)

    def has_next(self) -> bool:
        return len(self._queue) > 0

    def get_next(self) -> RefBundle:
        return self._queue.popleft()