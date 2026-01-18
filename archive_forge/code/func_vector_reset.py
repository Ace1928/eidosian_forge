import gymnasium as gym
import numpy as np
from typing import Optional
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.utils.annotations import override
@override(VectorEnv)
def vector_reset(self, *, seeds=None, options=None):
    seeds = seeds or [None]
    options = options or [None]
    obs, infos = self.env.reset(seed=seeds[0], options=options[0])
    return ([obs for _ in range(self.num_envs)], [infos for _ in range(self.num_envs)])