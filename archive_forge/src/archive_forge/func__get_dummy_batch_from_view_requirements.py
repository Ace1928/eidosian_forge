from gymnasium.spaces import Box
import numpy as np
import random
import tree  # pip install dm_tree
from typing import (
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelWeights, TensorStructType, TensorType
@override(Policy)
def _get_dummy_batch_from_view_requirements(self, batch_size: int=1):
    return SampleBatch({SampleBatch.OBS: tree.map_structure(lambda s: s[None], self.observation_space.sample())})