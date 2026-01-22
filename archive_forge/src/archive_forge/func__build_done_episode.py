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
def _build_done_episode(self, env_id: EnvID, is_done: bool, outputs: List[SampleBatchType]):
    """Builds a MultiAgentSampleBatch from the episode and adds it to outputs.

        Args:
            env_id: The env id.
            is_done: Whether the env is done.
            outputs: The list of outputs to add the
        """
    episode: EpisodeV2 = self._active_episodes[env_id]
    batch_builder = self._batch_builders[env_id]
    episode.postprocess_episode(batch_builder=batch_builder, is_done=is_done, check_dones=is_done)
    if not self._multiple_episodes_in_batch:
        ma_sample_batch = _build_multi_agent_batch(episode.episode_id, batch_builder, self._large_batch_threshold, self._multiple_episodes_in_batch)
        if ma_sample_batch:
            outputs.append(ma_sample_batch)
        del self._batch_builders[env_id]