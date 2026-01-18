from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils import EzPickle
import numpy as np
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
def viewer_setup(self):
    self.viewer.cam.distance = self.model.stat.extent * 0.5