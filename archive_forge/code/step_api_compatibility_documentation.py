import gym
from gym.logger import deprecation
from gym.utils.step_api_compatibility import (
Steps through the environment, returning 5 or 4 items depending on `apply_step_compatibility`.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info) or (observation, reward, done, info)
        