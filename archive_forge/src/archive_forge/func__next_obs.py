import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, Tuple
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.utils.spaces.space_utils import flatten_space
def _next_obs(self):
    self.current_obs = self.observation_space.sample()
    self.current_obs_flattened = tree.flatten(self.current_obs)
    return self.current_obs