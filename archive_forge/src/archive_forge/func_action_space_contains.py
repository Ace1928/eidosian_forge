import gymnasium as gym
import logging
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import (
from ray.rllib.utils.typing import (
from ray.util import log_once
@override(MultiAgentEnv)
def action_space_contains(self, x: MultiAgentDict) -> bool:
    if not isinstance(x, dict):
        return False
    return all((self.action_space.contains(val) for val in x.values()))