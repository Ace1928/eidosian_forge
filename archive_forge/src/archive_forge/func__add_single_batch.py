from enum import Enum
import logging
import numpy as np
import random
from typing import Any, Dict, List, Optional, Union
import ray  # noqa F401
import psutil
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
from ray.rllib.utils.actor_manager import FaultAwareApply
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.metrics.window_stat import WindowStat
from ray.rllib.utils.replay_buffers.base import ReplayBufferInterface
from ray.rllib.utils.typing import SampleBatchType
from ray.util.annotations import DeveloperAPI
from ray.util.debug import log_once
@DeveloperAPI
def _add_single_batch(self, item: SampleBatchType, **kwargs) -> None:
    """Add a SampleBatch of experiences to self._storage.

        An item consists of either one or more timesteps, a sequence or an
        episode. Differs from add() in that it does not consider the storage
        unit or type of batch and simply stores it.

        Args:
            item: The batch to be added.
            ``**kwargs``: Forward compatibility kwargs.
        """
    self._num_timesteps_added += item.count
    self._num_timesteps_added_wrap += item.count
    if self._next_idx >= len(self._storage):
        self._storage.append(item)
        self._est_size_bytes += item.size_bytes()
    else:
        item_to_be_removed = self._storage[self._next_idx]
        self._est_size_bytes -= item_to_be_removed.size_bytes()
        self._storage[self._next_idx] = item
        self._est_size_bytes += item.size_bytes()
    if self._eviction_started:
        self._evicted_hit_stats.push(self._hit_count[self._next_idx])
        self._hit_count[self._next_idx] = 0
    if self._num_timesteps_added_wrap >= self.capacity:
        self._eviction_started = True
        self._num_timesteps_added_wrap = 0
        self._next_idx = 0
    else:
        self._next_idx += 1