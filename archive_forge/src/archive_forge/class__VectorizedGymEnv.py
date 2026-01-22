import logging
import gymnasium as gym
import numpy as np
from typing import Callable, List, Optional, Tuple, Union, Set
from ray.rllib.env.base_env import BaseEnv, _DUMMY_AGENT_ID
from ray.rllib.utils.annotations import Deprecated, override, PublicAPI
from ray.rllib.utils.typing import (
from ray.util import log_once
class _VectorizedGymEnv(VectorEnv):
    """Internal wrapper to translate any gym.Envs into a VectorEnv object."""

    def __init__(self, make_env: Optional[Callable[[int], EnvType]]=None, existing_envs: Optional[List[gym.Env]]=None, num_envs: int=1, *, observation_space: Optional[gym.Space]=None, action_space: Optional[gym.Space]=None, restart_failed_sub_environments: bool=False, env_config=None, policy_config=None):
        """Initializes a _VectorizedGymEnv object.

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
                a vectorized env) throws any error during env stepping, we will try to
                restart the faulty sub-environment. This is done
                without disturbing the other (still intact) sub-environments.
        """
        self.envs = existing_envs
        self.make_env = make_env
        self.restart_failed_sub_environments = restart_failed_sub_environments
        while len(self.envs) < num_envs:
            self.envs.append(make_env(len(self.envs)))
        super().__init__(observation_space=observation_space or self.envs[0].observation_space, action_space=action_space or self.envs[0].action_space, num_envs=num_envs)

    @override(VectorEnv)
    def vector_reset(self, *, seeds: Optional[List[int]]=None, options: Optional[List[dict]]=None) -> Tuple[List[EnvObsType], List[EnvInfoDict]]:
        seeds = seeds or [None] * self.num_envs
        options = options or [None] * self.num_envs
        resetted_obs = []
        resetted_infos = []
        for i in range(len(self.envs)):
            while True:
                obs, infos = self.reset_at(i, seed=seeds[i], options=options[i])
                if not isinstance(obs, Exception):
                    break
            resetted_obs.append(obs)
            resetted_infos.append(infos)
        return (resetted_obs, resetted_infos)

    @override(VectorEnv)
    def reset_at(self, index: Optional[int]=None, *, seed: Optional[int]=None, options: Optional[dict]=None) -> Tuple[Union[EnvObsType, Exception], Union[EnvInfoDict, Exception]]:
        if index is None:
            index = 0
        try:
            obs_and_infos = self.envs[index].reset(seed=seed, options=options)
        except Exception as e:
            if self.restart_failed_sub_environments:
                logger.exception(e.args[0])
                self.restart_at(index)
                obs_and_infos = (e, {})
            else:
                raise e
        return obs_and_infos

    @override(VectorEnv)
    def restart_at(self, index: Optional[int]=None) -> None:
        if index is None:
            index = 0
        try:
            self.envs[index].close()
        except Exception as e:
            if log_once('close_sub_env'):
                logger.warning(f'Trying to close old and replaced sub-environment (at vector index={index}), but closing resulted in error:\n{e}')
        logger.warning(f'Trying to restart sub-environment at index {index}.')
        self.envs[index] = self.make_env(index)
        logger.warning(f'Sub-environment at index {index} restarted successfully.')

    @override(VectorEnv)
    def vector_step(self, actions: List[EnvActionType]) -> Tuple[List[EnvObsType], List[float], List[bool], List[bool], List[EnvInfoDict]]:
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = ([], [], [], [], [])
        for i in range(self.num_envs):
            try:
                results = self.envs[i].step(actions[i])
            except Exception as e:
                if self.restart_failed_sub_environments:
                    logger.exception(e.args[0])
                    self.restart_at(i)
                    results = (e, 0.0, True, True, {})
                else:
                    raise e
            obs, reward, terminated, truncated, info = results
            if not isinstance(info, dict):
                raise ValueError('Info should be a dict, got {} ({})'.format(info, type(info)))
            obs_batch.append(obs)
            reward_batch.append(reward)
            terminated_batch.append(terminated)
            truncated_batch.append(truncated)
            info_batch.append(info)
        return (obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch)

    @override(VectorEnv)
    def get_sub_environments(self) -> List[EnvType]:
        return self.envs

    @override(VectorEnv)
    def try_render_at(self, index: Optional[int]=None):
        if index is None:
            index = 0
        return self.envs[index].render()