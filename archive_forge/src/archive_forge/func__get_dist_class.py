import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import logging
import tree  # pip install dm_tree
from typing import Dict, List, Optional, Tuple, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.sac.sac_tf_policy import (
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.policy.torch_mixins import TargetNetworkMixin
from ray.rllib.utils.torch_utils import (
from ray.rllib.utils.typing import (
def _get_dist_class(policy: Policy, config: AlgorithmConfigDict, action_space: gym.spaces.Space) -> Type[TorchDistributionWrapper]:
    """Helper function to return a dist class based on config and action space.

    Args:
        policy: The policy for which to return the action
            dist class.
        config: The Algorithm's config dict.
        action_space (gym.spaces.Space): The action space used.

    Returns:
        Type[TFActionDistribution]: A TF distribution class.
    """
    if hasattr(policy, 'dist_class') and policy.dist_class is not None:
        return policy.dist_class
    elif config['model'].get('custom_action_dist'):
        action_dist_class, _ = ModelCatalog.get_action_dist(action_space, config['model'], framework='torch')
        return action_dist_class
    elif isinstance(action_space, Discrete):
        return TorchCategorical
    elif isinstance(action_space, Simplex):
        return TorchDirichlet
    else:
        assert isinstance(action_space, Box)
        if config['normalize_actions']:
            return TorchSquashedGaussian if not config['_use_beta_distribution'] else TorchBeta
        else:
            return TorchDiagGaussian