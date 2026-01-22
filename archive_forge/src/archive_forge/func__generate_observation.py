import copy
import gymnasium as gym
import logging
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import override
from typing import Dict, Optional
from ray.rllib.examples.env.utils.interfaces import InfoAccumulationInterface
def _generate_observation(self):
    obs = np.zeros((self.grid_size, self.grid_size, 4))
    obs[self.red_pos[0], self.red_pos[1], 0] = 1
    obs[self.blue_pos[0], self.blue_pos[1], 1] = 1
    if self.red_coin:
        obs[self.coin_pos[0], self.coin_pos[1], 2] = 1
    else:
        obs[self.coin_pos[0], self.coin_pos[1], 3] = 1
    obs = self._get_obs_invariant_to_the_player_trained(obs)
    return obs