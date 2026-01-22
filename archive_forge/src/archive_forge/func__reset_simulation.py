from os import path
from typing import Optional, Union
import numpy as np
import gym
from gym import error, logger, spaces
from gym.spaces import Space
def _reset_simulation(self):
    mujoco.mj_resetData(self.model, self.data)