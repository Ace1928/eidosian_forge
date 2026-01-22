from gymnasium.spaces import Box, MultiDiscrete, Tuple as TupleSpace
import logging
import numpy as np
import random
import time
from typing import Callable, Optional, Tuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import MultiAgentDict, PolicyID, AgentID
def _get_step_results(self):
    """Collects those agents' obs/rewards that have to act in next `step`.

        Returns:
            Tuple:
                obs: Multi-agent observation dict.
                    Only those observations for which to get new actions are
                    returned.
                rewards: Rewards dict matching `obs`.
                dones: Done dict with only an __all__ multi-agent entry in it.
                    __all__=True, if episode is done for all agents.
                infos: An (empty) info dict.
        """
    obs = {}
    rewards = {}
    infos = {}
    for behavior_name in self.unity_env.behavior_specs:
        decision_steps, terminal_steps = self.unity_env.get_steps(behavior_name)
        for agent_id, idx in decision_steps.agent_id_to_index.items():
            key = behavior_name + '_{}'.format(agent_id)
            os = tuple((o[idx] for o in decision_steps.obs))
            os = os[0] if len(os) == 1 else os
            obs[key] = os
            rewards[key] = decision_steps.reward[idx] + decision_steps.group_reward[idx]
        for agent_id, idx in terminal_steps.agent_id_to_index.items():
            key = behavior_name + '_{}'.format(agent_id)
            if key not in obs:
                os = tuple((o[idx] for o in terminal_steps.obs))
                obs[key] = os = os[0] if len(os) == 1 else os
            rewards[key] = terminal_steps.reward[idx] + terminal_steps.group_reward[idx]
    return (obs, rewards, {'__all__': False}, {'__all__': False}, infos)