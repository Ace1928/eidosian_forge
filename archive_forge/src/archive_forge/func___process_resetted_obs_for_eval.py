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
def __process_resetted_obs_for_eval(self, env_id: EnvID, obs: Dict[EnvID, Dict[AgentID, EnvObsType]], infos: Dict[EnvID, Dict[AgentID, EnvInfoDict]], episode: EpisodeV2, to_eval: Dict[PolicyID, List[AgentConnectorDataType]]):
    """Process resetted obs through agent connectors for policy eval.

        Args:
            env_id: The env id.
            obs: The Resetted obs.
            episode: New episode.
            to_eval: List of agent connector data for policy eval.
        """
    per_policy_resetted_obs: Dict[PolicyID, List] = defaultdict(list)
    for agent_id, raw_obs in obs[env_id].items():
        policy_id: PolicyID = episode.policy_for(agent_id)
        per_policy_resetted_obs[policy_id].append((agent_id, raw_obs))
    for policy_id, agents_obs in per_policy_resetted_obs.items():
        policy = self._worker.policy_map[policy_id]
        acd_list: List[AgentConnectorDataType] = [AgentConnectorDataType(env_id, agent_id, {SampleBatch.NEXT_OBS: obs, SampleBatch.INFOS: infos, SampleBatch.T: episode.length, SampleBatch.AGENT_INDEX: episode.agent_index(agent_id)}) for agent_id, obs in agents_obs]
        processed = policy.agent_connectors(acd_list)
        for d in processed:
            episode.add_init_obs(agent_id=d.agent_id, init_obs=d.data.raw_dict[SampleBatch.NEXT_OBS], init_infos=d.data.raw_dict[SampleBatch.INFOS], t=d.data.raw_dict[SampleBatch.T])
            to_eval[policy_id].append(d)