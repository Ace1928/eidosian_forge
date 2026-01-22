import copy
import logging
import math
import os
import sys
from typing import (
from packaging import version
import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.learner.learner import LearnerHyperparameters
from ray.rllib.core.learner.learner_group_config import LearnerGroupConfig, ModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import ModuleID, SingleAgentRLModuleSpec
from ray.rllib.core.learner.learner import TorchCompileWhatToCompile
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.atari_wrappers import is_atari
from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils import deep_update, merge_dicts
from ray.rllib.utils.annotations import (
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import NotProvided, from_config
from ray.rllib.utils.gym import (
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.serialization import (
from ray.rllib.utils.torch_utils import TORCH_COMPILE_REQUIRED_VERSION
from ray.rllib.utils.typing import (
from ray.tune.logger import Logger
from ray.tune.registry import get_trainable_cls
from ray.tune.result import TRIAL_INFO
from ray.tune.tune import _Config
@staticmethod
def _serialize_dict(config):
    config['callbacks'] = serialize_type(config['callbacks'])
    config['sample_collector'] = serialize_type(config['sample_collector'])
    if isinstance(config['env'], type):
        config['env'] = serialize_type(config['env'])
    if 'replay_buffer_config' in config and isinstance(config['replay_buffer_config'].get('type'), type):
        config['replay_buffer_config']['type'] = serialize_type(config['replay_buffer_config']['type'])
    if isinstance(config['exploration_config'].get('type'), type):
        config['exploration_config']['type'] = serialize_type(config['exploration_config']['type'])
    if isinstance(config['model'].get('custom_model'), type):
        config['model']['custom_model'] = serialize_type(config['model']['custom_model'])
    ma_config = config.get('multiagent')
    if ma_config is not None:
        if isinstance(ma_config.get('policies'), (set, tuple)):
            ma_config['policies'] = list(ma_config['policies'])
        if ma_config.get('policy_mapping_fn'):
            ma_config['policy_mapping_fn'] = NOT_SERIALIZABLE
        if ma_config.get('policies_to_train'):
            ma_config['policies_to_train'] = NOT_SERIALIZABLE
    if isinstance(config.get('policies'), (set, tuple)):
        config['policies'] = list(config['policies'])
    if config.get('policy_mapping_fn'):
        config['policy_mapping_fn'] = NOT_SERIALIZABLE
    if config.get('policies_to_train'):
        config['policies_to_train'] = NOT_SERIALIZABLE
    return config