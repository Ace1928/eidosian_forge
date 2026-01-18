import gymnasium as gym
import numpy as np
from ray.rllib.utils.annotations import PublicAPI
Represents a variable-length list of child spaces.

    Example:
        self.observation_space = spaces.Repeated(spaces.Box(4,), max_len=10)
            --> from 0 to 10 boxes of shape (4,)

    See also: documentation for rllib.models.RepeatedValues, which shows how
        the lists are represented as batched input for ModelV2 classes.
    