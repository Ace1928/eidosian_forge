import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import random
class WheelBanditEnv(gym.Env):
    """Wheel bandit environment for 2D contexts
    (see https://arxiv.org/abs/1802.09127).
    """
    DEFAULT_CONFIG_WHEEL = {'delta': 0.5, 'mu_1': 1.2, 'mu_2': 1, 'mu_3': 50, 'std': 0.01}
    feature_dim = 2
    num_actions = 5

    def __init__(self, config=None):
        self.config = copy.copy(self.DEFAULT_CONFIG_WHEEL)
        if config is not None and type(config) == dict:
            self.config.update(config)
        self.delta = self.config['delta']
        self.mu_1 = self.config['mu_1']
        self.mu_2 = self.config['mu_2']
        self.mu_3 = self.config['mu_3']
        self.std = self.config['std']
        self.action_space = Discrete(self.num_actions)
        self.observation_space = Box(low=-1, high=1, shape=(self.feature_dim,))
        self.means = [self.mu_1] + 4 * [self.mu_2]
        self._elapsed_steps = 0
        self._current_context = None

    def _sample_context(self):
        while True:
            state = np.random.uniform(-1, 1, self.feature_dim)
            if np.linalg.norm(state) <= 1:
                return state

    def reset(self, *, seed=None, options=None):
        self._current_context = self._sample_context()
        return (self._current_context, {})

    def step(self, action):
        assert self._elapsed_steps is not None, 'Cannot call env.step() before calling reset()'
        action = int(action)
        self._elapsed_steps += 1
        rewards = [np.random.normal(self.means[j], self.std) for j in range(self.num_actions)]
        context = self._current_context
        r_big = np.random.normal(self.mu_3, self.std)
        if np.linalg.norm(context) >= self.delta:
            if context[0] > 0:
                if context[1] > 0:
                    rewards[1] = r_big
                    opt_action = 1
                else:
                    rewards[4] = r_big
                    opt_action = 4
            elif context[1] > 0:
                rewards[2] = r_big
                opt_action = 2
            else:
                rewards[3] = r_big
                opt_action = 3
        else:
            opt_action = 0
        reward = rewards[action]
        regret = rewards[opt_action] - reward
        self._current_context = self._sample_context()
        return (self._current_context, reward, True, False, {'regret': regret, 'opt_action': opt_action})

    def render(self, mode='human'):
        raise NotImplementedError