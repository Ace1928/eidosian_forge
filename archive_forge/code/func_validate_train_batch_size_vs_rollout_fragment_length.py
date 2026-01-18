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
def validate_train_batch_size_vs_rollout_fragment_length(self) -> None:
    """Detects mismatches for `train_batch_size` vs `rollout_fragment_length`.

        Only applicable for algorithms, whose train_batch_size should be directly
        dependent on rollout_fragment_length (synchronous sampling, on-policy PG algos).

        If rollout_fragment_length != "auto", makes sure that the product of
        `rollout_fragment_length` x `num_rollout_workers` x `num_envs_per_worker`
        roughly (10%) matches the provided `train_batch_size`. Otherwise, errors with
        asking the user to set rollout_fragment_length to `auto` or to a matching
        value.

        Also, only checks this if `train_batch_size` > 0 (DDPPO sets this
        to -1 to auto-calculate the actual batch size later).

        Raises:
            ValueError: If there is a mismatch between user provided
            `rollout_fragment_length` and `train_batch_size`.
        """
    if self.rollout_fragment_length != 'auto' and (not self.in_evaluation) and (self.train_batch_size > 0):
        min_batch_size = max(self.num_rollout_workers, 1) * self.num_envs_per_worker * self.rollout_fragment_length
        batch_size = min_batch_size
        while batch_size < self.train_batch_size:
            batch_size += min_batch_size
        if batch_size - self.train_batch_size > 0.1 * self.train_batch_size or batch_size - min_batch_size - self.train_batch_size > 0.1 * self.train_batch_size:
            suggested_rollout_fragment_length = self.train_batch_size // (self.num_envs_per_worker * (self.num_rollout_workers or 1))
            raise ValueError(f"Your desired `train_batch_size` ({self.train_batch_size}) or a value 10% off of that cannot be achieved with your other settings (num_rollout_workers={self.num_rollout_workers}; num_envs_per_worker={self.num_envs_per_worker}; rollout_fragment_length={self.rollout_fragment_length})! Try setting `rollout_fragment_length` to 'auto' OR {suggested_rollout_fragment_length}.")