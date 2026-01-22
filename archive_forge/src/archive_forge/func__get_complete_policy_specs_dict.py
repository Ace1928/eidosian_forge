import copy
import importlib.util
import logging
import os
import platform
import threading
from collections import defaultdict
from types import FunctionType
from typing import (
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Discrete, MultiDiscrete, Space
import ray
from ray import ObjectRef
from ray import cloudpickle as pickle
from ray.rllib.connectors.util import (
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.base_env import BaseEnv, convert_to_base_env
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.atari_wrappers import is_atari, wrap_deepmind
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.evaluation.sampler import SyncSampler
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.offline import (
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import (
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils import check_env, force_list
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.debug import summarize, update_global_seed_if_necessary
from ray.rllib.utils.deprecation import DEPRECATED_VALUE, deprecation_warning
from ray.rllib.utils.error import ERR_MSG_NO_GPUS, HOWTO_CHANGE_CONFIG
from ray.rllib.utils.filter import Filter, NoFilter, get_filter
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.policy import create_policy_for_framework, validate_policy_id
from ray.rllib.utils.sgd import do_minibatch_sgd
from ray.rllib.utils.tf_run_builder import _TFRunBuilder
from ray.rllib.utils.tf_utils import get_gpu_devices as get_tf_gpu_devices
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
from ray.rllib.utils.typing import (
from ray.tune.registry import registry_contains_input, registry_get_input
from ray.util.annotations import PublicAPI
from ray.util.debug import disable_log_once_globally, enable_periodic_logging, log_once
from ray.util.iter import ParallelIteratorWorker
def _get_complete_policy_specs_dict(self, policy_dict: MultiAgentPolicyConfigDict) -> MultiAgentPolicyConfigDict:
    """Processes the policy dict and creates a new copy with the processed attrs.

        This processes the observation_space and prepares them for passing to rl module
        construction. It also merges the policy configs with the algorithm config.
        During this processing, we will also construct the preprocessors dict.
        """
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    updated_policy_dict = copy.deepcopy(policy_dict)
    self.preprocessors = self.preprocessors or {}
    for name, policy_spec in sorted(updated_policy_dict.items()):
        logger.debug('Creating policy for {}'.format(name))
        if isinstance(policy_spec.config, AlgorithmConfig):
            merged_conf = policy_spec.config
        else:
            merged_conf: 'AlgorithmConfig' = self.config.copy(copy_frozen=False)
            merged_conf.update_from_dict(policy_spec.config or {})
        merged_conf.worker_index = self.worker_index
        obs_space = policy_spec.observation_space
        self.preprocessors[name] = None
        if self.preprocessing_enabled:
            preprocessor = ModelCatalog.get_preprocessor_for_space(obs_space, merged_conf.model, include_multi_binary=self.config.get('_enable_new_api_stack', False))
            if preprocessor is not None:
                obs_space = preprocessor.observation_space
            if not merged_conf.enable_connectors:
                self.preprocessors[name] = preprocessor
        policy_spec.config = merged_conf
        policy_spec.observation_space = obs_space
    return updated_policy_dict