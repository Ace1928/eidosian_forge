from contextlib import closing
from io import StringIO
from os import path
from typing import Optional
import numpy as np
from gym import Env, logger, spaces
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled
def _limit_coordinates(self, coord: np.ndarray) -> np.ndarray:
    """Prevent the agent from falling out of the grid world."""
    coord[0] = min(coord[0], self.shape[0] - 1)
    coord[0] = max(coord[0], 0)
    coord[1] = min(coord[1], self.shape[1] - 1)
    coord[1] = max(coord[1], 0)
    return coord