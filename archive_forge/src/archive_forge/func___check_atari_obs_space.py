import gymnasium as gym
import logging
import numpy as np
import re
from typing import (
import tree  # pip install dm_tree
import ray.cloudpickle as pickle
from ray.rllib.models.preprocessors import ATARI_OBS_SHAPE
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import (
from ray.util import log_once
from ray.util.annotations import PublicAPI
def __check_atari_obs_space(obs):
    if any((o.shape == ATARI_OBS_SHAPE if isinstance(o, np.ndarray) else False for o in tree.flatten(obs))):
        if log_once('warn_about_possibly_non_wrapped_atari_env'):
            logger.warning('The observation you fed into local_policy_inference() has dimensions (210, 160, 3), which is the standard for atari environments. If RLlib raises an error including a related dimensionality mismatch, you may need to use ray.rllib.env.wrappers.atari_wrappers.wrap_deepmind to wrap you environment.')