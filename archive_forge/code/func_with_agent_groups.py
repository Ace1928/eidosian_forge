import gymnasium as gym
import logging
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import (
from ray.rllib.utils.typing import (
from ray.util import log_once
def with_agent_groups(self, groups: Dict[str, List[AgentID]], obs_space: gym.Space=None, act_space: gym.Space=None) -> 'MultiAgentEnv':
    """Convenience method for grouping together agents in this env.

        An agent group is a list of agent IDs that are mapped to a single
        logical agent. All agents of the group must act at the same time in the
        environment. The grouped agent exposes Tuple action and observation
        spaces that are the concatenated action and obs spaces of the
        individual agents.

        The rewards of all the agents in a group are summed. The individual
        agent rewards are available under the "individual_rewards" key of the
        group info return.

        Agent grouping is required to leverage algorithms such as Q-Mix.

        Args:
            groups: Mapping from group id to a list of the agent ids
                of group members. If an agent id is not present in any group
                value, it will be left ungrouped. The group id becomes a new agent ID
                in the final environment.
            obs_space: Optional observation space for the grouped
                env. Must be a tuple space. If not provided, will infer this to be a
                Tuple of n individual agents spaces (n=num agents in a group).
            act_space: Optional action space for the grouped env.
                Must be a tuple space. If not provided, will infer this to be a Tuple
                of n individual agents spaces (n=num agents in a group).

        .. testcode::
            :skipif: True

            from ray.rllib.env.multi_agent_env import MultiAgentEnv
            class MyMultiAgentEnv(MultiAgentEnv):
                # define your env here
                ...
            env = MyMultiAgentEnv(...)
            grouped_env = env.with_agent_groups(env, {
              "group1": ["agent1", "agent2", "agent3"],
              "group2": ["agent4", "agent5"],
            })

        """
    from ray.rllib.env.wrappers.group_agents_wrapper import GroupAgentsWrapper
    return GroupAgentsWrapper(self, groups, obs_space, act_space)