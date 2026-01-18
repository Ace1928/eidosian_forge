import logging
import gymnasium as gym
import numpy as np
from typing import Callable, List, Optional, Tuple, Union, Set
from ray.rllib.env.base_env import BaseEnv, _DUMMY_AGENT_ID
from ray.rllib.utils.annotations import Deprecated, override, PublicAPI
from ray.rllib.utils.typing import (
from ray.util import log_once
@staticmethod
def vectorize_gym_envs(make_env: Optional[Callable[[int], EnvType]]=None, existing_envs: Optional[List[gym.Env]]=None, num_envs: int=1, action_space: Optional[gym.Space]=None, observation_space: Optional[gym.Space]=None, restart_failed_sub_environments: bool=False, env_config=None, policy_config=None) -> '_VectorizedGymEnv':
    """Translates any given gym.Env(s) into a VectorizedEnv object.

        Args:
            make_env: Factory that produces a new gym.Env taking the sub-env's
                vector index as only arg. Must be defined if the
                number of `existing_envs` is less than `num_envs`.
            existing_envs: Optional list of already instantiated sub
                environments.
            num_envs: Total number of sub environments in this VectorEnv.
            action_space: The action space. If None, use existing_envs[0]'s
                action space.
            observation_space: The observation space. If None, use
                existing_envs[0]'s observation space.
            restart_failed_sub_environments: If True and any sub-environment (within
                a vectorized env) throws any error during env stepping, the
                Sampler will try to restart the faulty sub-environment. This is done
                without disturbing the other (still intact) sub-environment and without
                the RolloutWorker crashing.

        Returns:
            The resulting _VectorizedGymEnv object (subclass of VectorEnv).
        """
    return _VectorizedGymEnv(make_env=make_env, existing_envs=existing_envs or [], num_envs=num_envs, observation_space=observation_space, action_space=action_space, restart_failed_sub_environments=restart_failed_sub_environments)