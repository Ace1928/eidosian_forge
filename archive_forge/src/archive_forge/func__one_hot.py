import logging
from typing import List, Optional, Type, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.multi_agent_env_compatibility import (
from ray.rllib.utils.error import (
from ray.rllib.utils.gym import check_old_gym_env
from ray.rllib.utils.numpy import one_hot, one_hot_multidiscrete
from ray.rllib.utils.spaces.space_utils import (
from ray.util import log_once
from ray.util.annotations import PublicAPI
def _one_hot(self, data, space_struct):
    if space_struct is None:
        raise ValueError(f'Cannot `one_hot` data in `{type(self).__name__}` if a gym.Space was NOT provided during construction!')

    def _convert(dat_, space):
        if isinstance(space, gym.spaces.Discrete):
            return one_hot(dat_, depth=space.n)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return one_hot_multidiscrete(dat_, depths=space.nvec)
        return dat_
    if isinstance(data, list):
        data = [tree.map_structure(_convert, dslice, space_struct) for dslice in data]
    else:
        data = tree.map_structure(_convert, data, space_struct)
    return data