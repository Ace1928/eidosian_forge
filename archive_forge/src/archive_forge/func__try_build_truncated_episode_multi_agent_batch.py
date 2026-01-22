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
def _try_build_truncated_episode_multi_agent_batch(self, batch_builder: _PolicyCollectorGroup, episode: EpisodeV2) -> Union[None, SampleBatch, MultiAgentBatch]:
    if self._count_steps_by == 'env_steps':
        built_steps = batch_builder.env_steps
        ongoing_steps = episode.active_env_steps
    else:
        built_steps = batch_builder.agent_steps
        ongoing_steps = episode.active_agent_steps
    if built_steps + ongoing_steps >= self._rollout_fragment_length:
        if self._count_steps_by != 'agent_steps':
            assert built_steps + ongoing_steps == self._rollout_fragment_length, f'built_steps ({built_steps}) + ongoing_steps ({ongoing_steps}) != rollout_fragment_length ({self._rollout_fragment_length}).'
        if built_steps < self._rollout_fragment_length:
            episode.postprocess_episode(batch_builder=batch_builder, is_done=False)
        if batch_builder.agent_steps > 0:
            return _build_multi_agent_batch(episode.episode_id, batch_builder, self._large_batch_threshold, self._multiple_episodes_in_batch)
        elif log_once('no_agent_steps'):
            logger.warning('Your environment seems to be stepping w/o ever emitting agent observations (agents are never requested to act)!')
    return None