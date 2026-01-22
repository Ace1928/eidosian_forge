import math
import warnings
from typing import TYPE_CHECKING, Optional
import numpy as np
import gym
from gym import error, spaces
from gym.error import DependencyNotInstalled
from gym.utils import EzPickle, colorize
from gym.utils.step_api_compatibility import step_api_compatibility
class LunarLanderContinuous:

    def __init__(self):
        raise error.Error('Error initializing LunarLanderContinuous Environment.\nCurrently, we do not support initializing this mode of environment by calling the class directly.\nTo use this environment, instead create it by specifying the continuous keyword in gym.make, i.e.\ngym.make("LunarLander-v2", continuous=True)')