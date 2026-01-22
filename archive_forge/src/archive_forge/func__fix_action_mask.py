from gymnasium.spaces import Box, Dict, Discrete
import numpy as np
from ray.rllib.examples.env.random_env import RandomEnv
def _fix_action_mask(self, obs):
    self.valid_actions = np.round(obs['action_mask'])
    obs['action_mask'] = self.valid_actions