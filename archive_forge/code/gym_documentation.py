import gymnasium as gym
from typing import Optional
from ray.util.annotations import DeveloperAPI
Converts an old gym (NOT gymnasium) Space into a gymnasium.Space.

    Args:
        space: The gym.Space to convert to gymnasium.Space.

    Returns:
         The converted gymnasium.space object.
    