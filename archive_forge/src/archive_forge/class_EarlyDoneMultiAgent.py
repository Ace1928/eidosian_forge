import gymnasium as gym
import numpy as np
import random
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
from ray.rllib.examples.env.mock_env import MockEnv, MockEnv2
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
from ray.rllib.utils.deprecation import Deprecated
class EarlyDoneMultiAgent(MultiAgentEnv):
    """Env for testing when the env terminates (after agent 0 does)."""

    def __init__(self):
        super().__init__()
        self.agents = [MockEnv(3), MockEnv(5)]
        self._agent_ids = set(range(len(self.agents)))
        self.terminateds = set()
        self.truncateds = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_terminated = {}
        self.last_truncated = {}
        self.last_info = {}
        self.i = 0
        self.observation_space = gym.spaces.Discrete(10)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        self.terminateds = set()
        self.truncateds = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_terminated = {}
        self.last_truncated = {}
        self.last_info = {}
        self.i = 0
        for i, a in enumerate(self.agents):
            self.last_obs[i], self.last_info[i] = a.reset()
            self.last_rew[i] = 0
            self.last_terminated[i] = False
            self.last_truncated[i] = False
        obs_dict = {self.i: self.last_obs[self.i]}
        info_dict = {self.i: self.last_info[self.i]}
        self.i = (self.i + 1) % len(self.agents)
        return (obs_dict, info_dict)

    def step(self, action_dict):
        assert len(self.terminateds) != len(self.agents)
        for i, action in action_dict.items():
            self.last_obs[i], self.last_rew[i], self.last_terminated[i], self.last_truncated[i], self.last_info[i] = self.agents[i].step(action)
        obs = {self.i: self.last_obs[self.i]}
        rew = {self.i: self.last_rew[self.i]}
        terminated = {self.i: self.last_terminated[self.i]}
        truncated = {self.i: self.last_truncated[self.i]}
        info = {self.i: self.last_info[self.i]}
        if terminated[self.i]:
            rew[self.i] = 0
            self.terminateds.add(self.i)
        if truncated[self.i]:
            rew[self.i] = 0
            self.truncateds.add(self.i)
        self.i = (self.i + 1) % len(self.agents)
        terminated['__all__'] = len(self.terminateds) == len(self.agents) - 1
        truncated['__all__'] = len(self.truncateds) == len(self.agents) - 1
        return (obs, rew, terminated, truncated, info)