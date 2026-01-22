import sys
from typing import (
import numpy as np
from gym import spaces
from gym.logger import warn
from gym.utils import seeding
class RewardWrapper(Wrapper):
    """Superclass of wrappers that can modify the returning reward from a step.

    If you would like to apply a function to the reward that is returned by the base environment before
    passing it to learning code, you can simply inherit from :class:`RewardWrapper` and overwrite the method
    :meth:`reward` to implement that transformation.
    This transformation might change the reward range; to specify the reward range of your wrapper,
    you can simply define :attr:`self.reward_range` in :meth:`__init__`.

    Let us look at an example: Sometimes (especially when we do not have control over the reward
    because it is intrinsic), we want to clip the reward to a range to gain some numerical stability.
    To do that, we could, for instance, implement the following wrapper::

        class ClipReward(gym.RewardWrapper):
            def __init__(self, env, min_reward, max_reward):
                super().__init__(env)
                self.min_reward = min_reward
                self.max_reward = max_reward
                self.reward_range = (min_reward, max_reward)

            def reward(self, reward):
                return np.clip(reward, self.min_reward, self.max_reward)
    """

    def step(self, action):
        """Modifies the reward using :meth:`self.reward` after the environment :meth:`env.step`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (observation, self.reward(reward), terminated, truncated, info)

    def reward(self, reward):
        """Returns a modified ``reward``."""
        raise NotImplementedError