import copy
from collections import Iterable
import numpy as np
from numba import jit, prange
from numba.typed import List
from ray.rllib.examples.env.coin_game_non_vectorized_env import CoinGame
from ray.rllib.utils import override
def _optional_unvectorize(self, obs, rewards=None):
    if self.batch_size == 1 and (not self.force_vectorized):
        obs = [one_obs[0, ...] for one_obs in obs]
        if rewards is not None:
            rewards[0], rewards[1] = (rewards[0][0], rewards[1][0])
    return (obs, rewards)