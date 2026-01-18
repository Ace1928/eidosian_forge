import math
from typing import Optional, Union
import numpy as np
import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.error import DependencyNotInstalled, InvalidAction
from gym.utils import EzPickle
def vertical_ind(place, val):
    return [(place * s, H - (h + h * val)), ((place + 1) * s, H - (h + h * val)), ((place + 1) * s, H - h), ((place + 0) * s, H - h)]