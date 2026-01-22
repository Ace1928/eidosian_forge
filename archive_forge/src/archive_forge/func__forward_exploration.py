from typing import Mapping, Any
from ray.rllib.core.rl_module import RLModule
from ray.rllib.policy.sample_batch import SampleBatch
import tree
import pathlib
import gymnasium as gym
def _forward_exploration(self, batch, **kwargs):
    return self._random_forward(batch, **kwargs)