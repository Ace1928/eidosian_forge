import gymnasium as gym
import queue
import threading
import uuid
from typing import Callable, Tuple, Optional, TYPE_CHECKING
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import (
from ray.rllib.utils.deprecation import deprecation_warning
@PublicAPI
class ExternalEnvWrapper(BaseEnv):
    """Internal adapter of ExternalEnv to BaseEnv."""

    def __init__(self, external_env: 'ExternalEnv', preprocessor: 'Preprocessor'=None):
        from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
        self.external_env = external_env
        self.prep = preprocessor
        self.multiagent = issubclass(type(external_env), ExternalMultiAgentEnv)
        self._action_space = external_env.action_space
        if preprocessor:
            self._observation_space = preprocessor.observation_space
        else:
            self._observation_space = external_env.observation_space
        external_env.start()

    @override(BaseEnv)
    def poll(self) -> Tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict]:
        with self.external_env._results_avail_condition:
            results = self._poll()
            while len(results[0]) == 0:
                self.external_env._results_avail_condition.wait()
                results = self._poll()
                if not self.external_env.is_alive():
                    raise Exception('Serving thread has stopped.')
        return results

    @override(BaseEnv)
    def send_actions(self, action_dict: MultiEnvDict) -> None:
        from ray.rllib.env.base_env import _DUMMY_AGENT_ID
        if self.multiagent:
            for env_id, actions in action_dict.items():
                self.external_env._episodes[env_id].action_queue.put(actions)
        else:
            for env_id, action in action_dict.items():
                self.external_env._episodes[env_id].action_queue.put(action[_DUMMY_AGENT_ID])

    def _poll(self) -> Tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict]:
        from ray.rllib.env.base_env import with_dummy_agent_id
        all_obs, all_rewards, all_terminateds, all_truncateds, all_infos = ({}, {}, {}, {}, {})
        off_policy_actions = {}
        for eid, episode in self.external_env._episodes.copy().items():
            data = episode.get_data()
            cur_terminated = episode.cur_terminated_dict['__all__'] if self.multiagent else episode.cur_terminated
            cur_truncated = episode.cur_truncated_dict['__all__'] if self.multiagent else episode.cur_truncated
            if cur_terminated or cur_truncated:
                del self.external_env._episodes[eid]
            if data:
                if self.prep:
                    all_obs[eid] = self.prep.transform(data['obs'])
                else:
                    all_obs[eid] = data['obs']
                all_rewards[eid] = data['reward']
                all_terminateds[eid] = data['terminated']
                all_truncateds[eid] = data['truncated']
                all_infos[eid] = data['info']
                if 'off_policy_action' in data:
                    off_policy_actions[eid] = data['off_policy_action']
        if self.multiagent:
            for eid, eid_dict in all_obs.items():
                for agent_id in eid_dict.keys():

                    def fix(d, zero_val):
                        if agent_id not in d[eid]:
                            d[eid][agent_id] = zero_val
                    fix(all_rewards, 0.0)
                    fix(all_terminateds, False)
                    fix(all_truncateds, False)
                    fix(all_infos, {})
            return (all_obs, all_rewards, all_terminateds, all_truncateds, all_infos, off_policy_actions)
        else:
            return (with_dummy_agent_id(all_obs), with_dummy_agent_id(all_rewards), with_dummy_agent_id(all_terminateds, '__all__'), with_dummy_agent_id(all_truncateds, '__all__'), with_dummy_agent_id(all_infos), with_dummy_agent_id(off_policy_actions))

    @property
    @override(BaseEnv)
    @PublicAPI
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    @property
    @override(BaseEnv)
    @PublicAPI
    def action_space(self) -> gym.Space:
        return self._action_space