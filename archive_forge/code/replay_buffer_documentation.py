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
Fetches concatenated samples at given indices from the storage.