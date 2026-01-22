import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import random
def _sample_context(self):
    while True:
        state = np.random.uniform(-1, 1, self.feature_dim)
        if np.linalg.norm(state) <= 1:
            return state