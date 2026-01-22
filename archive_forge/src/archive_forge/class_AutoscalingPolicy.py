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
class AutoscalingPolicy:
    """Autoscaling policy for an actor pool, determining when the pool should be scaled
    up and when it should be scaled down.
    """

    def __init__(self, autoscaling_config: 'AutoscalingConfig'):
        self._config = autoscaling_config

    @property
    def min_workers(self) -> int:
        """The minimum number of actors that must be in the actor pool."""
        return self._config.min_workers

    @property
    def max_workers(self) -> int:
        """The maximum number of actors that can be added to the actor pool."""
        return self._config.max_workers

    def should_scale_up(self, num_total_workers: int, num_running_workers: int) -> bool:
        """Whether the actor pool should scale up by adding a new actor.

        Args:
            num_total_workers: Total number of workers in actor pool.
            num_running_workers: Number of currently running workers in actor pool.

        Returns:
            Whether the actor pool should be scaled up by one actor.
        """
        if num_total_workers < self._config.min_workers:
            return True
        else:
            return num_total_workers < self._config.max_workers and num_total_workers > 0 and (num_running_workers / num_total_workers > self._config.ready_to_total_workers_ratio)

    def should_scale_down(self, num_total_workers: int, num_idle_workers: int) -> bool:
        """Whether the actor pool should scale down by terminating an inactive actor.

        Args:
            num_total_workers: Total number of workers in actor pool.
            num_idle_workers: Number of currently idle workers in the actor pool.

        Returns:
            Whether the actor pool should be scaled down by one actor.
        """
        return num_total_workers > self._config.min_workers and num_idle_workers / num_total_workers > self._config.idle_to_total_workers_ratio