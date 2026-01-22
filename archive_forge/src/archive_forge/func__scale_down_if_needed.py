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
def _scale_down_if_needed(self):
    """Try to scale down the pool if the autoscaling policy allows it."""
    self._kill_inactive_workers_if_done()
    while self._autoscaling_policy.should_scale_down(num_total_workers=self._actor_pool.num_total_actors(), num_idle_workers=self._actor_pool.num_idle_actors()):
        killed = self._actor_pool.kill_inactive_actor()
        if not killed:
            break