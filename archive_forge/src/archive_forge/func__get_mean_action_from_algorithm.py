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
def _get_mean_action_from_algorithm(alg: 'Algorithm', obs: np.ndarray) -> np.ndarray:
    """Returns the mean action computed by the given algorithm.

    Note: This makes calls to `Algorithm.compute_single_action`

    Args:
        alg: The constructed algorithm to run inference on.
        obs: The observation to compute the action for.

    Returns:
        The mean action computed by the algorithm over 5000 samples.

    """
    out = []
    for _ in range(5000):
        out.append(float(alg.compute_single_action(obs)))
    return np.mean(out)