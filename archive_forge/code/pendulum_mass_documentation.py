from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.utils import EzPickle
import numpy as np
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

        Returns:
            float: The current mass of the pendulum (self.m in the PendulumEnv
                object).
        