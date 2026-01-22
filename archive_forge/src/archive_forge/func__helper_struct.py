import gymnasium as gym
from gymnasium.spaces import Tuple, Dict
import numpy as np
from ray.rllib.utils.annotations import DeveloperAPI
import tree  # pip install dm_tree
from typing import Any, List, Optional, Union
def _helper_struct(space_):
    if isinstance(space_, Tuple):
        return tuple((_helper_struct(s) for s in space_))
    elif isinstance(space_, Dict):
        return {k: _helper_struct(space_[k]) for k in space_.spaces}
    else:
        return space_