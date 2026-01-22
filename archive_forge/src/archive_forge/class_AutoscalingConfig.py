import collections
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union
import ray
from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.map_operator import MapOperator, _map_task
from ray.data._internal.execution.operators.map_transformer import MapTransformer
from ray.data._internal.execution.util import locality_string
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
@dataclass
class AutoscalingConfig:
    """Configuration for an autoscaling actor pool."""
    min_workers: int
    max_workers: int
    max_tasks_in_flight: int = DEFAULT_MAX_TASKS_IN_FLIGHT
    ready_to_total_workers_ratio: float = 0.8
    idle_to_total_workers_ratio: float = 0.5

    def __post_init__(self):
        if self.min_workers < 1:
            raise ValueError('min_workers must be >= 1, got: ', self.min_workers)
        if self.max_workers is not None and self.min_workers > self.max_workers:
            raise ValueError('min_workers must be <= max_workers, got: ', self.min_workers, self.max_workers)
        if self.max_tasks_in_flight < 1:
            raise ValueError('max_tasks_in_flight must be >= 1, got: ', self.max_tasks_in_flight)

    @classmethod
    def from_compute_strategy(cls, compute_strategy: ActorPoolStrategy):
        """Convert a legacy ActorPoolStrategy to an AutoscalingConfig."""
        assert isinstance(compute_strategy, ActorPoolStrategy)
        return cls(min_workers=compute_strategy.min_size, max_workers=compute_strategy.max_size, max_tasks_in_flight=compute_strategy.max_tasks_in_flight_per_actor or DEFAULT_MAX_TASKS_IN_FLIGHT, ready_to_total_workers_ratio=compute_strategy.ready_to_total_workers_ratio)