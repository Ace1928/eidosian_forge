import argparse
import gymnasium as gym
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import get_trainable_cls
class NonVectorizedEnvToBeVectorizedIntoRemoteBaseEnv(TaskSettableEnv):
    """Class for a single sub-env to be vectorized into RemoteBaseEnv.

    If you specify this class directly under the "env" config key, RLlib
    will auto-wrap

    Note that you may implement your own custom APIs. Here, we demonstrate
    using RLlib's TaskSettableEnv API (which is a simple sub-class
    of gym.Env).
    """

    def __init__(self, config=None):
        super().__init__()
        self.action_space = gym.spaces.Box(0, 1, shape=(1,))
        self.observation_space = gym.spaces.Box(0, 1, shape=(2,))
        self.task = 1

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        return (self.observation_space.sample(), {})

    def step(self, action):
        self.steps += 1
        done = truncated = self.steps > 10
        return (self.observation_space.sample(), 0, done, truncated, {})

    def set_task(self, task) -> None:
        """We can set the task of each sub-env (ray actor)"""
        print('Task set to {}'.format(task))
        self.task = task