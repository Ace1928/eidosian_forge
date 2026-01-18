import gymnasium as gym
import logging
from typing import Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import ray
from ray.util import log_once
from ray.rllib.env.base_env import BaseEnv, _DUMMY_AGENT_ID, ASYNC_RESET_RETURN
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EnvType, MultiEnvDict
Re-creates a sub-environment at the new index.