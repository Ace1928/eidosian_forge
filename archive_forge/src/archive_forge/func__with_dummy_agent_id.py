import logging
from typing import Callable, Tuple, Optional, List, Dict, Any, TYPE_CHECKING, Union, Set
import gymnasium as gym
import ray
from ray.rllib.utils.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EnvType, MultiEnvDict
@Deprecated(new='with_dummy_agent_id()', error=True)
def _with_dummy_agent_id(env_id_to_values: Dict[EnvID, Any], dummy_id: 'AgentID'=_DUMMY_AGENT_ID) -> MultiEnvDict:
    return {k: {dummy_id: v} for k, v in env_id_to_values.items()}