import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import random
class LinearDiscreteEnv(gym.Env):
    """Samples data from linearly parameterized arms.

    The reward for context X and arm i is given by X^T * theta_i, for some
    latent set of parameters {theta_i : i = 1, ..., k}.
    The thetas are sampled uniformly at random, the contexts are Gaussian,
    and Gaussian noise is added to the rewards.
    """
    DEFAULT_CONFIG_LINEAR = {'feature_dim': 8, 'num_actions': 4, 'reward_noise_std': 0.01}

    def __init__(self, config=None):
        self.config = copy.copy(self.DEFAULT_CONFIG_LINEAR)
        if config is not None and type(config) == dict:
            self.config.update(config)
        self.feature_dim = self.config['feature_dim']
        self.num_actions = self.config['num_actions']
        self.sigma = self.config['reward_noise_std']
        self.action_space = Discrete(self.num_actions)
        self.observation_space = Box(low=-10, high=10, shape=(self.feature_dim,))
        self.thetas = np.random.uniform(-1, 1, (self.num_actions, self.feature_dim))
        self.thetas /= np.linalg.norm(self.thetas, axis=1, keepdims=True)
        self._elapsed_steps = 0
        self._current_context = None

    def _sample_context(self):
        return np.random.normal(scale=1 / 3, size=(self.feature_dim,))

    def reset(self, *, seed=None, options=None):
        self._current_context = self._sample_context()
        return (self._current_context, {})

    def step(self, action):
        assert self._elapsed_steps is not None, 'Cannot call env.step() beforecalling reset()'
        assert action < self.num_actions, 'Invalid action.'
        action = int(action)
        context = self._current_context
        rewards = self.thetas.dot(context)
        opt_action = rewards.argmax()
        regret = rewards.max() - rewards[action]
        rewards += np.random.normal(scale=self.sigma, size=rewards.shape)
        reward = rewards[action]
        self._current_context = self._sample_context()
        return (self._current_context, reward, True, False, {'regret': regret, 'opt_action': opt_action})

    def render(self, mode='human'):
        raise NotImplementedError