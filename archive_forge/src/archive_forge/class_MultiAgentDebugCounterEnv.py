import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
class MultiAgentDebugCounterEnv(MultiAgentEnv):

    def __init__(self, config):
        super().__init__()
        self.num_agents = config['num_agents']
        self.base_episode_len = config.get('base_episode_len', 103)
        self.action_space = gym.spaces.Box(-float('inf'), float('inf'), shape=(2,))
        self.observation_space = gym.spaces.Box(float('-inf'), float('inf'), (4,))
        self.timesteps = [0] * self.num_agents
        self.terminateds = set()
        self.truncateds = set()
        self._skip_env_checking = True

    def reset(self, *, seed=None, options=None):
        self.timesteps = [0] * self.num_agents
        self.terminateds = set()
        self.truncateds = set()
        return ({i: np.array([i, 0.0, 0.0, 0.0], dtype=np.float32) for i in range(self.num_agents)}, {})

    def step(self, action_dict):
        obs, rew, terminated, truncated = ({}, {}, {}, {})
        for i, action in action_dict.items():
            self.timesteps[i] += 1
            obs[i] = np.array([i, action[0], action[1], self.timesteps[i]])
            rew[i] = self.timesteps[i] % 3
            terminated[i] = False
            truncated[i] = True if self.timesteps[i] > self.base_episode_len + i else False
            if terminated[i]:
                self.terminateds.add(i)
            if truncated[i]:
                self.truncateds.add(i)
        terminated['__all__'] = len(self.terminateds) == self.num_agents
        truncated['__all__'] = len(self.truncateds) == self.num_agents
        return (obs, rew, terminated, truncated, {})