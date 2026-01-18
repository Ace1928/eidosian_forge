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
@PublicAPI
def try_import_pyspiel(error: bool=False):
    """Tries importing pyspiel and returns the module (or None).

    Args:
        error: Whether to raise an error if pyspiel cannot be imported.

    Returns:
        The pyspiel module.

    Raises:
        ImportError: If error=True and pyspiel is not installed.
    """
    try:
        import pyspiel
        return pyspiel
    except ImportError:
        if error:
            raise ImportError('Could not import pyspiel! Pygame is not a dependency of RLlib and RLlib requires you to install pygame separately: `pip install pygame`.')
        return None