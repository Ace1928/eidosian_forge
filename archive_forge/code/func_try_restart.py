import gymnasium as gym
import logging
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import (
from ray.rllib.utils.typing import (
from ray.util import log_once
@override(BaseEnv)
def try_restart(self, env_id: Optional[EnvID]=None) -> None:
    if isinstance(env_id, int):
        env_id = [env_id]
    if env_id is None:
        env_id = list(range(len(self.envs)))
    for idx in env_id:
        logger.warning(f'Trying to restart sub-environment at index {idx}.')
        self.env_states[idx].env = self.envs[idx] = self.make_env(idx)
        logger.warning(f'Sub-environment at index {idx} restarted successfully.')