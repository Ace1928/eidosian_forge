from collections import defaultdict
import logging
import time
import tree  # pip install dm_tree
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, Tuple, Union
import numpy as np
from ray.rllib.env.base_env import ASYNC_RESET_RETURN, BaseEnv
from ray.rllib.env.external_env import ExternalEnvWrapper
from ray.rllib.env.wrappers.atari_wrappers import MonitorEnv, get_wrapper_by_cls
from ray.rllib.evaluation.collectors.simple_list_collector import _PolicyCollectorGroup
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch, concat_samples
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.filter import Filter
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import unbatch, get_original_space
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
class _PerfStats:
    """Sampler perf stats that will be included in rollout metrics."""

    def __init__(self, ema_coef: Optional[float]=None):
        self.ema_coef = ema_coef
        self.iters = 0
        self.raw_obs_processing_time = 0.0
        self.inference_time = 0.0
        self.action_processing_time = 0.0
        self.env_wait_time = 0.0
        self.env_render_time = 0.0

    def incr(self, field: str, value: Union[int, float]):
        if field == 'iters':
            self.iters += value
            return
        if self.ema_coef is None:
            self.__dict__[field] += value
        else:
            self.__dict__[field] = (1.0 - self.ema_coef) * self.__dict__[field] + self.ema_coef * value

    def _get_avg(self):
        factor = MS_TO_SEC / self.iters
        return {'mean_raw_obs_processing_ms': self.raw_obs_processing_time * factor, 'mean_inference_ms': self.inference_time * factor, 'mean_action_processing_ms': self.action_processing_time * factor, 'mean_env_wait_ms': self.env_wait_time * factor, 'mean_env_render_ms': self.env_render_time * factor}

    def _get_ema(self):
        return {'mean_raw_obs_processing_ms': self.raw_obs_processing_time * MS_TO_SEC, 'mean_inference_ms': self.inference_time * MS_TO_SEC, 'mean_action_processing_ms': self.action_processing_time * MS_TO_SEC, 'mean_env_wait_ms': self.env_wait_time * MS_TO_SEC, 'mean_env_render_ms': self.env_render_time * MS_TO_SEC}

    def get(self):
        if self.ema_coef is None:
            return self._get_avg()
        else:
            return self._get_ema()