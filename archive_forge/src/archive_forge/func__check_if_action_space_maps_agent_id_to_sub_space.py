import gymnasium as gym
import logging
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import (
from ray.rllib.utils.typing import (
from ray.util import log_once
@DeveloperAPI
def _check_if_action_space_maps_agent_id_to_sub_space(self) -> bool:
    """Checks if action space maps from agent ids to spaces of individual agents."""
    return hasattr(self, 'action_space') and isinstance(self.action_space, gym.spaces.Dict) and (set(self.action_space.keys()) == self.get_agent_ids())