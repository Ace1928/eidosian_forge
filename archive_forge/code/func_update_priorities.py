import numpy as np
from typing import Any, Dict, Optional
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer, StorageUnit
from ray.rllib.utils.typing import SampleBatchType
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def update_priorities(self, *args, **kwargs) -> None:
    """Update priorities of items at given indices.

        No-op for this replay buffer.

        Args:
            ``*args``   : Forward compatibility args.
            ``**kwargs``: Forward compatibility kwargs.
        """
    pass