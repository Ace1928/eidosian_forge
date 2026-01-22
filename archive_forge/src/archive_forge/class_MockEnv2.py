import gymnasium as gym
import numpy as np
from typing import Optional
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.utils.annotations import override
class MockEnv2(gym.Env):
    """Mock environment for testing purposes.

    Observation=ts (discrete space!), reward=100.0, episode-len is
    configurable. Actions are ignored.
    """
    metadata = {'render.modes': ['rgb_array']}
    render_mode: Optional[str] = 'rgb_array'

    def __init__(self, episode_length):
        self.episode_length = episode_length
        self.i = 0
        self.observation_space = gym.spaces.Discrete(self.episode_length + 1)
        self.action_space = gym.spaces.Discrete(2)
        self.rng_seed = None

    def reset(self, *, seed=None, options=None):
        self.i = 0
        if seed is not None:
            self.rng_seed = seed
        return (self.i, {})

    def step(self, action):
        self.i += 1
        terminated = truncated = self.i >= self.episode_length
        return (self.i, 100.0, terminated, truncated, {})

    def render(self):
        return np.random.randint(0, 256, size=(300, 400, 3), dtype=np.uint8)