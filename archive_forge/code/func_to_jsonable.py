import gymnasium as gym
import numpy as np
from ray.rllib.utils.annotations import PublicAPI
def to_jsonable(self, sample_n):
    return np.array(sample_n).tolist()