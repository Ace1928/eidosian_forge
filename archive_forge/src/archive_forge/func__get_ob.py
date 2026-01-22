from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Union
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.images import rgb2gray, resize
def _get_ob(self):
    assert len(self.frames) == self.k
    return np.concatenate(self.frames, axis=2)