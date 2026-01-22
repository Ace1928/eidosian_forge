from typing import Mapping, Any
from ray.rllib.core.rl_module import RLModule
from ray.rllib.policy.sample_batch import SampleBatch
import tree
import pathlib
import gymnasium as gym
def _random_forward(self, batch, **kwargs):
    obs_batch_size = len(tree.flatten(batch[SampleBatch.OBS])[0])
    actions = [self.config.action_space.sample() for _ in range(obs_batch_size)]
    return {SampleBatch.ACTIONS: actions}