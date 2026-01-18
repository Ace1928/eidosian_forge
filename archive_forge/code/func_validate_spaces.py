import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from functools import partial
import logging
from typing import Dict, List, Optional, Tuple, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.dqn.dqn_tf_policy import (
from ray.rllib.algorithms.sac.sac_tf_model import SACTFModel
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import (
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import TargetNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import get_variable, try_import_tf
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.tf_utils import huber_loss, make_tf_callable
from ray.rllib.utils.typing import (
def validate_spaces(policy: Policy, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> None:
    """Validates the observation- and action spaces used for the Policy.

    Args:
        policy: The policy, whose spaces are being validated.
        observation_space (gym.spaces.Space): The observation space to
            validate.
        action_space (gym.spaces.Space): The action space to validate.
        config: The Policy's config dict.

    Raises:
        UnsupportedSpaceException: If one of the spaces is not supported.
    """
    if not isinstance(action_space, (Box, Discrete, Simplex)):
        raise UnsupportedSpaceException('Action space ({}) of {} is not supported for SAC. Must be [Box|Discrete|Simplex].'.format(action_space, policy))
    elif isinstance(action_space, (Box, Simplex)) and len(action_space.shape) > 1:
        raise UnsupportedSpaceException('Action space ({}) of {} has multiple dimensions {}. '.format(action_space, policy, action_space.shape) + 'Consider reshaping this into a single dimension, using a Tuple action space, or the multi-agent API.')