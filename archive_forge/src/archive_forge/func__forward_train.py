from typing import Mapping, Any
from ray.rllib.core.rl_module import RLModule
from ray.rllib.policy.sample_batch import SampleBatch
import tree
import pathlib
import gymnasium as gym
def _forward_train(self, *args, **kwargs):
    raise NotImplementedError('This RLM should only run in evaluation.')