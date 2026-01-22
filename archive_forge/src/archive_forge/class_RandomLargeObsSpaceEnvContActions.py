import copy
import gymnasium as gym
from gymnasium.spaces import Discrete, Tuple
import numpy as np
from ray.rllib.examples.env.multi_agent import make_multi_agent
class RandomLargeObsSpaceEnvContActions(RandomEnv):

    def __init__(self, config=None):
        config = config or {}
        config.update({'observation_space': gym.spaces.Box(-1.0, 1.0, (5000,)), 'action_space': gym.spaces.Box(-1.0, 1.0, (5,))})
        super().__init__(config=config)