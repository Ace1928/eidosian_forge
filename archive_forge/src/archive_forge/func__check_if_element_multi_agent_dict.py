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
def _check_if_element_multi_agent_dict(env, element, function_string, base_env=False):
    if not isinstance(element, dict):
        if base_env:
            error = f'The element returned by {function_string} contains values that are not MultiAgentDicts. Instead, they are of type: {type(element)}'
        else:
            error = f'The element returned by {function_string} is not a MultiAgentDict. Instead, it is of type:  {type(element)}'
        raise ValueError(error)
    agent_ids: Set = copy(env.get_agent_ids())
    agent_ids.add('__all__')
    if not all((k in agent_ids for k in element)):
        if base_env:
            error = f'The element returned by {function_string} has agent_ids that are not the names of the agents in the env.agent_ids in this\nMultiEnvDict: {list(element.keys())}\nAgent_ids in this env:{list(env.get_agent_ids())}'
        else:
            error = f'The element returned by {function_string} has agent_ids that are not the names of the agents in the env. \nAgent_ids in this MultiAgentDict: {list(element.keys())}\nAgent_ids in this env:{list(env.get_agent_ids())}. You likely need to add the private attribute `_agent_ids` to your env, which is a set containing the ids of agents supported by your env.'
        raise ValueError(error)