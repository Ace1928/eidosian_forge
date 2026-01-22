import gymnasium as gym
from gymnasium.spaces import Tuple, Dict
import numpy as np
from ray.rllib.utils.annotations import DeveloperAPI
import tree  # pip install dm_tree
from typing import Any, List, Optional, Union
def _helper_flatten(space_, return_list):
    from ray.rllib.utils.spaces.flexdict import FlexDict
    if isinstance(space_, Tuple):
        for s in space_:
            _helper_flatten(s, return_list)
    elif isinstance(space_, (Dict, FlexDict)):
        for k in sorted(space_.spaces):
            _helper_flatten(space_[k], return_list)
    else:
        return_list.append(space_)