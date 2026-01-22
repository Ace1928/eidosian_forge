from gymnasium.spaces import Dict, Discrete, MultiDiscrete, Tuple
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
def _obs(self):
    if self.with_state:
        return {self.agent_1: {'obs': self.agent_1_obs(), ENV_STATE: self.state}, self.agent_2: {'obs': self.agent_2_obs(), ENV_STATE: self.state}}
    else:
        return {self.agent_1: self.agent_1_obs(), self.agent_2: self.agent_2_obs()}