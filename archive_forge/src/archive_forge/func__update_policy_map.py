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
def _update_policy_map(self, *, policy_dict: MultiAgentPolicyConfigDict, policy: Optional[Policy]=None, policy_states: Optional[Dict[PolicyID, PolicyState]]=None, single_agent_rl_module_spec: Optional[SingleAgentRLModuleSpec]=None) -> None:
    """Updates the policy map (and other stuff) on this worker.

        It performs the following:
            1. It updates the observation preprocessors and updates the policy_specs
                with the postprocessed observation_spaces.
            2. It updates the policy_specs with the complete algorithm_config (merged
                with the policy_spec's config).
            3. If needed it will update the self.marl_module_spec on this worker
            3. It updates the policy map with the new policies
            4. It updates the filter dict
            5. It calls the on_create_policy() hook of the callbacks on the newly added
                policies.

        Args:
            policy_dict: The policy dict to update the policy map with.
            policy: The policy to update the policy map with.
            policy_states: The policy states to update the policy map with.
            single_agent_rl_module_spec: The SingleAgentRLModuleSpec to add to the
                MultiAgentRLModuleSpec. If None, the config's
                `get_default_rl_module_spec` method's output will be used to create
                the policy with.
        """
    updated_policy_dict = self._get_complete_policy_specs_dict(policy_dict)
    if self.config._enable_new_api_stack:
        spec = self.config.get_marl_module_spec(policy_dict=updated_policy_dict, single_agent_rl_module_spec=single_agent_rl_module_spec)
        if self.marl_module_spec is None:
            self.marl_module_spec = spec
        else:
            self.marl_module_spec.add_modules(spec.module_specs)
        updated_policy_dict = self._update_policy_dict_with_marl_module(updated_policy_dict)
    self._build_policy_map(policy_dict=updated_policy_dict, policy=policy, policy_states=policy_states)
    self._update_filter_dict(updated_policy_dict)
    if policy is None:
        self._call_callbacks_on_create_policy()
    if self.worker_index == 0:
        logger.info(f'Built policy map: {self.policy_map}')
        logger.info(f'Built preprocessor map: {self.preprocessors}')