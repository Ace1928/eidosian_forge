import logging
from typing import List, Optional, Type, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.multi_agent_env_compatibility import (
from ray.rllib.utils.error import (
from ray.rllib.utils.gym import check_old_gym_env
from ray.rllib.utils.numpy import one_hot, one_hot_multidiscrete
from ray.rllib.utils.spaces.space_utils import (
from ray.util import log_once
from ray.util.annotations import PublicAPI
def _gym_env_creator(env_context: EnvContext, env_descriptor: Union[str, Type[gym.Env]], auto_wrap_old_gym_envs: bool=True) -> gym.Env:
    """Tries to create a gym env given an EnvContext object and descriptor.

    Note: This function tries to construct the env from a string descriptor
    only using possibly installed RL env packages (such as gym, pybullet_envs,
    vizdoomgym, etc..). These packages are no installation requirements for
    RLlib. In case you would like to support more such env packages, add the
    necessary imports and construction logic below.

    Args:
        env_context: The env context object to configure the env.
            Note that this is a config dict, plus the properties:
            `worker_index`, `vector_index`, and `remote`.
        env_descriptor: The env descriptor as a gym-registered string, e.g. CartPole-v1,
            ALE/MsPacman-v5, VizdoomBasic-v0, or CartPoleContinuousBulletEnv-v0.
            Alternatively, the gym.Env subclass to use.
        auto_wrap_old_gym_envs: Whether to auto-wrap old gym environments (using
            the pre 0.24 gym APIs, e.g. reset() returning single obs and no info
            dict). If True, RLlib will automatically wrap the given gym env class
            with the gym-provided compatibility wrapper (gym.wrappers.EnvCompatibility).
            If False, RLlib will produce a descriptive error on which steps to perform
            to upgrade to gymnasium (or to switch this flag to True).

    Returns:
        The actual gym environment object.

    Raises:
        gym.error.Error: If the env cannot be constructed.
    """
    try:
        import pybullet_envs
        pybullet_envs.getList()
    except (AttributeError, ModuleNotFoundError, ImportError):
        pass
    try:
        import vizdoomgym
        vizdoomgym.__name__
    except (ModuleNotFoundError, ImportError):
        pass
    try:
        if isinstance(env_descriptor, type):
            env = env_descriptor(env_context)
        else:
            env = gym.make(env_descriptor, **env_context)
        if auto_wrap_old_gym_envs:
            try:
                obs_and_infos = env.reset(seed=None, options={})
                check_old_gym_env(reset_results=obs_and_infos)
            except Exception:
                if log_once('auto_wrap_gym_api'):
                    logger.warning('`config.auto_wrap_old_gym_envs` is activated AND you seem to have provided an old gym-API environment. RLlib will therefore try to auto-fix the following error. However, please consider switching over to the new `gymnasium` APIs:\n' + ERR_MSG_OLD_GYM_API)
                if isinstance(env, MultiAgentEnv):
                    env = MultiAgentEnvCompatibility(env)
                else:
                    env = gym.wrappers.EnvCompatibility(env)
                obs_and_infos = env.reset(seed=None, options={})
                check_old_gym_env(reset_results=obs_and_infos)
    except gym.error.Error:
        raise EnvError(ERR_MSG_INVALID_ENV_DESCRIPTOR.format(env_descriptor))
    return env