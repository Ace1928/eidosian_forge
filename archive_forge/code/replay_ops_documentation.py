from typing import Optional
import random
from ray.rllib.utils.replay_buffers.replay_buffer import warn_replay_capacity
from ray.rllib.utils.typing import SampleBatchType
Initialize SimpleReplayBuffer.

        Args:
            num_slots: Number of batches to store in total.
        