import math
from typing import Optional, Union
import numpy as np
import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.error import DependencyNotInstalled, InvalidAction
from gym.utils import EzPickle
def _reinit_colors(self, randomize):
    assert self.domain_randomize, 'domain_randomize must be True to use this function.'
    if randomize:
        self.road_color = self.np_random.uniform(0, 210, size=3)
        self.bg_color = self.np_random.uniform(0, 210, size=3)
        self.grass_color = np.copy(self.bg_color)
        idx = self.np_random.integers(3)
        self.grass_color[idx] += 20