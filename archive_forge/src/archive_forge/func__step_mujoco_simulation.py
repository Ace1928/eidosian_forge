from os import path
from typing import Optional, Union
import numpy as np
import gym
from gym import error, logger, spaces
from gym.spaces import Space
def _step_mujoco_simulation(self, ctrl, n_frames):
    self.data.ctrl[:] = ctrl
    mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
    mujoco.mj_rnePostConstraint(self.model, self.data)