from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def _generate_single_agent_episode(self, agent_id: str, agent_episode_ids: Optional[Dict[str, str]]=None, observations: Optional[List[MultiAgentDict]]=None, actions: Optional[List[MultiAgentDict]]=None, rewards: Optional[List[MultiAgentDict]]=None, infos: Optional[List[MultiAgentDict]]=None, terminateds: Union[MultiAgentDict, bool]=False, truncateds: Union[MultiAgentDict, bool]=False, extra_model_outputs: Optional[List[MultiAgentDict]]=None) -> SingleAgentEpisode:
    """Generates a SingleAgentEpisode from multi-agent data.

        Note, if no data is provided an empty `SingleAgentEpiosde`
        will be returned that starts at `SingleAgentEpisode.t_started=0`.
        """
    episode_id = None if agent_episode_ids is None else agent_episode_ids[agent_id]
    if len(self.global_t_to_local_t) > 0:
        agent_observations = None if observations is None else self._get_single_agent_data(agent_id, observations, shift=-self.ts_carriage_return)
        agent_actions = None if actions is None else self._get_single_agent_data(agent_id, actions, use_global_t_to_local_t=False)
        agent_rewards = None if rewards is None else self._get_single_agent_data(agent_id, rewards, use_global_t_to_local_t=False)
        agent_infos = None if infos is None else self._get_single_agent_data(agent_id, infos, shift=-self.ts_carriage_return)
        _agent_extra_model_outputs = None if extra_model_outputs is None else self._get_single_agent_data(agent_id, extra_model_outputs, use_global_t_to_local_t=False)
        agent_extra_model_outputs = defaultdict(list)
        for _model_out in _agent_extra_model_outputs:
            for key, val in _model_out.items():
                agent_extra_model_outputs[key].append(val)
        agent_is_terminated = terminateds.get(agent_id, False)
        agent_is_truncated = truncateds.get(agent_id, False)
        if agent_actions and agent_observations and (len(agent_observations) == len(agent_actions)):
            if agent_extra_model_outputs:
                assert all((len(v) == len(agent_actions) for v in agent_extra_model_outputs.values())), f"Agent {agent_id} doesn't have the same number of `extra_model_outputs` as it has actions ({len(agent_actions)})."
                self.agent_buffers[agent_id]['extra_model_outputs'].get_nowait()
                self.agent_buffers[agent_id]['extra_model_outputs'].put_nowait({k: v.pop() for k, v in agent_extra_model_outputs.items()})
            self.agent_buffers[agent_id]['actions'].put_nowait(agent_actions.pop())
        if agent_rewards and observations:
            partial_agent_rewards_t = _IndexMapping()
            partial_agent_rewards = []
            agent_rewards = []
            agent_reward = 0.0
            for t, reward in enumerate(rewards):
                if agent_id in reward:
                    partial_agent_rewards.append(reward[agent_id])
                    agent_reward += reward[agent_id]
                    partial_agent_rewards_t.append(t + self.ts_carriage_return + 1)
                    if t + 1 in self.global_t_to_local_t[agent_id][1:]:
                        agent_rewards.append(agent_reward)
                        agent_reward = 0.0
            self.agent_buffers[agent_id]['rewards'].put_nowait(self.agent_buffers[agent_id]['rewards'].get_nowait() + agent_reward)
            self.partial_rewards_t[agent_id] = partial_agent_rewards_t
            self.partial_rewards[agent_id] = partial_agent_rewards
        return SingleAgentEpisode(id_=episode_id, observations=agent_observations, actions=agent_actions, rewards=agent_rewards, infos=agent_infos, terminated=agent_is_terminated, truncated=agent_is_truncated, extra_model_outputs=agent_extra_model_outputs)
    else:
        return SingleAgentEpisode(id_=episode_id)