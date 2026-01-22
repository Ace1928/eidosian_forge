import logging
import traceback
from copy import copy
from typing import TYPE_CHECKING, Optional, Set, Union
import numpy as np
import tree  # pip install dm_tree
from ray.actor import ActorHandle
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.error import ERR_MSG_OLD_GYM_API, UnsupportedSpaceException
from ray.rllib.utils.gym import check_old_gym_env, try_import_gymnasium_and_gym
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.typing import EnvType
from ray.util import log_once
def _check_if_vetor_env_list(env, element, function_string):
    if not isinstance(element, list) or len(element) != env.num_envs:
        raise ValueError(f'The element returned by {function_string} is not a list OR the length of the returned list is not the same as the number of sub-environments ({env.num_envs}) in your VectorEnv! Instead, your {function_string} returned {element}')