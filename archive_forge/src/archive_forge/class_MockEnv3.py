import gymnasium as gym
import numpy as np
from typing import Optional
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.utils.annotations import override
class MockEnv3(gym.Env):
    """Mock environment for testing purposes.

    Observation=ts (discrete space!), reward=100.0, episode-len is
    configurable. Actions are ignored.
    """

    def __init__(self, episode_length):
        self.episode_length = episode_length
        self.i = 0
        self.observation_space = gym.spaces.Discrete(100)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        self.i = 0
        return (self.i, {'timestep': 0})

    def step(self, action):
        self.i += 1
        terminated = truncated = self.i >= self.episode_length
        return (self.i, self.i, terminated, truncated, {'timestep': self.i})