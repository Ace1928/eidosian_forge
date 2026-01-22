from collections import OrderedDict
import gymnasium as gym
from typing import Dict, List, Optional
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.typing import AgentID
def _ungroup_items(self, items):
    out = {}
    for agent_id, value in items.items():
        if agent_id in self.groups:
            assert len(value) == len(self.groups[agent_id]), (agent_id, value, self.groups)
            for a, v in zip(self.groups[agent_id], value):
                out[a] = v
        else:
            out[agent_id] = value
    return out