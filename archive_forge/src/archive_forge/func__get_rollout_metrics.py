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
def _get_rollout_metrics(self, episode: EpisodeV2, policy_map: Dict[str, Policy]) -> List[RolloutMetrics]:
    """Get rollout metrics from completed episode."""
    atari_metrics: List[RolloutMetrics] = _fetch_atari_metrics(self._base_env)
    if atari_metrics is not None:
        for m in atari_metrics:
            m._replace(custom_metrics=episode.custom_metrics)
        return atari_metrics
    connector_metrics = {}
    active_agents = episode.get_agents()
    for agent in active_agents:
        policy_id = episode.policy_for(agent)
        policy = episode.policy_map[policy_id]
        connector_metrics[policy_id] = policy.get_connector_metrics()
    return [RolloutMetrics(episode_length=episode.length, episode_reward=episode.total_reward, agent_rewards=dict(episode.agent_rewards), custom_metrics=episode.custom_metrics, perf_stats={}, hist_data=episode.hist_data, media=episode.media, connector_metrics=connector_metrics)]