import numpy as np
import gymnasium as gym
class PendulumWrapper(gym.Wrapper):
    """Wrapper for the Pendulum-v1 environment.

    Adds an additional `reward` method for some model-based RL algos (e.g.
    MB-MPO).
    """
    _max_episode_steps = 200

    def __init__(self, **kwargs):
        env = gym.make('Pendulum-v1', **kwargs)
        gym.Wrapper.__init__(self, env)

    def reward(self, obs, action, obs_next):
        theta = np.arctan2(np.clip(obs[:, 1], -1.0, 1.0), np.clip(obs[:, 0], -1.0, 1.0))
        a = np.clip(action, -self.max_torque, self.max_torque)[0]
        costs = self.angle_normalize(theta) ** 2 + 0.1 * obs[:, 2] ** 2 + 0.001 * a ** 2
        return -costs

    @staticmethod
    def angle_normalize(x):
        return (x + np.pi) % (2 * np.pi) - np.pi