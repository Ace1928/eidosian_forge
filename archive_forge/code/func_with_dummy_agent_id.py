import logging
from typing import Callable, Tuple, Optional, List, Dict, Any, TYPE_CHECKING, Union, Set
import gymnasium as gym
import ray
from ray.rllib.utils.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EnvType, MultiEnvDict
@PublicAPI
def with_dummy_agent_id(env_id_to_values: Dict[EnvID, Any], dummy_id: 'AgentID'=_DUMMY_AGENT_ID) -> MultiEnvDict:
    ret = {}
    for env_id, value in env_id_to_values.items():
        ret[env_id] = value if isinstance(value, Exception) else {dummy_id: value}
    return ret