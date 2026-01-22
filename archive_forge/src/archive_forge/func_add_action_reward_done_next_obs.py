import random
from collections import defaultdict
import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.collectors.simple_list_collector import (
from ray.rllib.evaluation.collectors.agent_collector import AgentCollector
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.typing import AgentID, EnvID, EnvInfoDict, PolicyID, TensorType
def add_action_reward_done_next_obs(self, agent_id: AgentID, values: Dict[str, TensorType]) -> None:
    """Add action, reward, info, and next_obs as a new step.

        Args:
            agent_id: Agent ID.
            values: Dict of action, reward, info, and next_obs.
        """
    assert agent_id in self._agent_collectors
    self.active_agent_steps += 1
    self.total_agent_steps += 1
    if agent_id != _DUMMY_AGENT_ID:
        values['agent_id'] = agent_id
    self._agent_collectors[agent_id].add_action_reward_next_obs(values)
    reward = values[SampleBatch.REWARDS]
    self.total_reward += reward
    self.agent_rewards[agent_id, self.policy_for(agent_id)] += reward
    self._agent_reward_history[agent_id].append(reward)
    if SampleBatch.TERMINATEDS in values:
        self._last_terminateds[agent_id] = values[SampleBatch.TERMINATEDS]
    if SampleBatch.TRUNCATEDS in values:
        self._last_truncateds[agent_id] = values[SampleBatch.TRUNCATEDS]
    if SampleBatch.INFOS in values:
        self.set_last_info(agent_id, values[SampleBatch.INFOS])