import enum
import functools
from typing import Optional
import gymnasium as gym
import numpy as np
import tree
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from ray.rllib.core.models.base import Encoder
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models.distributions import Distribution
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
from ray.rllib.models.utils import get_filter_config
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import ViewRequirementsDict
from ray.rllib.utils.annotations import (
def _multi_categorical_dist_partial_helper(action_space: gym.Space, framework: str) -> Distribution:
    """Helper method to get a partial of a MultiCategorical Distribution.

    This is useful for when we want to create MultiCategorical Distribution from
    logits only (!) later, but know the action space now already.

    Args:
        action_space: The action space to get the child distribution classes for.
        framework: The framework to use.

    Returns:
        A partial of the MultiCategorical class.
    """
    if framework == 'torch':
        from ray.rllib.models.torch.torch_distributions import TorchMultiCategorical
        multi_categorical_dist_cls = TorchMultiCategorical
    elif framework == 'tf2':
        from ray.rllib.models.tf.tf_distributions import TfMultiCategorical
        multi_categorical_dist_cls = TfMultiCategorical
    else:
        raise ValueError(f'Unsupported framework: {framework}')
    partial_dist_cls = multi_categorical_dist_cls.get_partial_dist_cls(space=action_space, input_lens=list(action_space.nvec))
    return partial_dist_cls