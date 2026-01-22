from functools import partial
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
import logging
import numpy as np
import tree  # pip install dm_tree
from typing import List, Optional, Type, Union
from ray.tune.registry import (
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
from ray.rllib.models.tf.tf_action_dist import (
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.typing import ModelConfigDict, TensorType
@staticmethod
def _get_multi_action_distribution(dist_class, action_space, config, framework):
    if issubclass(dist_class, (MultiActionDistribution, TorchMultiActionDistribution)):
        flat_action_space = flatten_space(action_space)
        child_dists_and_in_lens = tree.map_structure(lambda s: ModelCatalog.get_action_dist(s, config, framework=framework), flat_action_space)
        child_dists = [e[0] for e in child_dists_and_in_lens]
        input_lens = [int(e[1]) for e in child_dists_and_in_lens]
        return (partial(dist_class, action_space=action_space, child_distributions=child_dists, input_lens=input_lens), int(sum(input_lens)))
    return (dist_class, dist_class.required_model_output_shape(action_space, config))