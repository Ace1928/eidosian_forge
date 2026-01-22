from collections import Counter
import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Tuple as GymTuple
import inspect
import logging
import numpy as np
import os
import pprint
import random
import re
import time
import tree  # pip install dm_tree
from typing import (
import yaml
import ray
from ray import air, tune
from ray.rllib.env.wrappers.atari_wrappers import is_atari, wrap_deepmind
from ray.rllib.utils.framework import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.tune import CLIReporter, run_experiments
def _do_check(alg, config, a_name, o_name):
    config_copy = config.copy()
    config_copy.validate()
    if config_copy._enable_new_api_stack:
        if o_name not in rlmodule_supported_observation_spaces:
            logger.warning('Skipping PPO test with RLModules for obs space {}'.format(o_name))
            return
        if a_name not in rlmodule_supported_action_spaces:
            logger.warning('Skipping PPO test with RLModules for action space {}'.format(a_name))
            return
    fw = config['framework']
    action_space = action_spaces_to_test[a_name]
    obs_space = observation_spaces_to_test[o_name]
    print('=== Testing {} (fw={}) action_space={} obs_space={} ==='.format(alg, fw, action_space, obs_space))
    t0 = time.time()
    config.update_from_dict(dict(env_config=dict(action_space=action_space, observation_space=obs_space, reward_space=Box(1.0, 1.0, shape=(), dtype=np.float32), p_terminated=1.0, check_action_bounds=check_bounds)))
    stat = 'ok'
    try:
        algo = config.build()
    except ray.exceptions.RayActorError as e:
        if len(e.args) >= 2 and isinstance(e.args[2], UnsupportedSpaceException):
            stat = 'unsupported'
        elif isinstance(e.args[0].args[2], UnsupportedSpaceException):
            stat = 'unsupported'
        else:
            raise
    except UnsupportedSpaceException:
        stat = 'unsupported'
    else:
        if alg not in ['SAC', 'PPO']:
            if o_name in ['atari', 'image']:
                if fw == 'torch':
                    assert isinstance(algo.get_policy().model, TorchVisionNet)
                else:
                    assert isinstance(algo.get_policy().model, VisionNet)
            elif o_name == 'continuous':
                if fw == 'torch':
                    assert isinstance(algo.get_policy().model, TorchFCNet)
                else:
                    assert isinstance(algo.get_policy().model, FCNet)
            elif o_name == 'vector2d':
                if fw == 'torch':
                    assert isinstance(algo.get_policy().model, (TorchComplexNet, TorchFCNet))
                else:
                    assert isinstance(algo.get_policy().model, (ComplexNet, FCNet))
        if train:
            algo.train()
        algo.stop()
    print('Test: {}, ran in {}s'.format(stat, time.time() - t0))