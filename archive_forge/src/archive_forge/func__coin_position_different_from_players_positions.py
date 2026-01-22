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
def _coin_position_different_from_players_positions(self):
    success = 0
    while success < self.NUM_AGENTS:
        self.coin_pos = self.np_random.integers(self.grid_size, size=2)
        success = 1 - self._same_pos(self.red_pos, self.coin_pos)
        success += 1 - self._same_pos(self.blue_pos, self.coin_pos)