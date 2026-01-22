import numpy as np
from typing import Any, Dict, Optional
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer, StorageUnit
from ray.rllib.utils.typing import SampleBatchType
from ray.util.annotations import DeveloperAPI
Restores all local state to the provided `state`.

        Args:
            state: The new state to set this buffer. Can be obtained by calling
            `self.get_state()`.
        