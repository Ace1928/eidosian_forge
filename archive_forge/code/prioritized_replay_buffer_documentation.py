import random
from typing import Any, Dict, List, Optional
import numpy as np
import ray  # noqa F401
import psutil  # noqa E402
from ray.rllib.execution.segment_tree import SumSegmentTree, MinSegmentTree
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics.window_stat import WindowStat
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer
from ray.rllib.utils.typing import SampleBatchType
from ray.util.annotations import DeveloperAPI
Restores all local state to the provided `state`.

        Args:
            state: The new state to set this buffer. Can be obtained by calling
            `self.get_state()`.
        