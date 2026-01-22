import json
import logging
import os
import platform
from abc import ABCMeta, abstractmethod
from typing import (
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Box
from packaging import version
import ray
import ray.cloudpickle as pickle
from ray.actor import ActorHandle
from ray.train import Checkpoint
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import (
from ray.rllib.utils.checkpoints import (
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.serialization import (
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.tensor_dtype import get_np_dtype
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
@PublicAPI
class PolicySpec:
    """A policy spec used in the "config.multiagent.policies" specification dict.

    As values (keys are the policy IDs (str)). E.g.:
    config:
        multiagent:
            policies: {
                "pol1": PolicySpec(None, Box, Discrete(2), {"lr": 0.0001}),
                "pol2": PolicySpec(config={"lr": 0.001}),
            }
    """

    def __init__(self, policy_class=None, observation_space=None, action_space=None, config=None):
        self.policy_class = policy_class
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config

    def __eq__(self, other: 'PolicySpec'):
        return self.policy_class == other.policy_class and self.observation_space == other.observation_space and (self.action_space == other.action_space) and (self.config == other.config)

    def serialize(self) -> Dict:
        from ray.rllib.algorithms.registry import get_policy_class_name
        cls = get_policy_class_name(self.policy_class)
        if cls is None:
            logger.warning(f'Can not figure out a durable policy name for {self.policy_class}. You are probably trying to checkpoint a custom policy. Raw policy class may cause problems when the checkpoint needs to be loaded in the future. To fix this, make sure you add your custom policy in rllib.algorithms.registry.POLICIES.')
            cls = self.policy_class
        return {'policy_class': cls, 'observation_space': space_to_dict(self.observation_space), 'action_space': space_to_dict(self.action_space), 'config': self.config}

    @classmethod
    def deserialize(cls, spec: Dict) -> 'PolicySpec':
        if isinstance(spec['policy_class'], str):
            from ray.rllib.algorithms.registry import get_policy_class
            policy_class = get_policy_class(spec['policy_class'])
        elif isinstance(spec['policy_class'], type):
            policy_class = spec['policy_class']
        else:
            raise AttributeError(f'Unknown policy class spec {spec['policy_class']}')
        return cls(policy_class=policy_class, observation_space=space_from_dict(spec['observation_space']), action_space=space_from_dict(spec['action_space']), config=spec['config'])