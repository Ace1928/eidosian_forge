import copy
from collections import Iterable
import numpy as np
from numba import jit, prange
from numba.typed import List
from ray.rllib.examples.env.coin_game_non_vectorized_env import CoinGame
from ray.rllib.utils import override
@jit(nopython=True)
def _unflatten_index(pos, grid_size):
    x_idx = pos % grid_size
    y_idx = pos // grid_size
    return np.array([y_idx, x_idx])