import random
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
def update_avail_actions(self):
    self.action_assignments = np.array([[0.0, 0.0]] * self.action_space.n, dtype=np.float32)
    self.action_mask = np.array([0.0] * self.action_space.n, dtype=np.int8)
    self.left_idx, self.right_idx = random.sample(range(self.action_space.n), 2)
    self.action_assignments[self.left_idx] = self.left_action_embed
    self.action_assignments[self.right_idx] = self.right_action_embed
    self.action_mask[self.left_idx] = 1
    self.action_mask[self.right_idx] = 1