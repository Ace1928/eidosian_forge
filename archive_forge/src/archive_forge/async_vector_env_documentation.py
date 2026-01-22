import multiprocessing as mp
import sys
import time
from copy import deepcopy
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import gym
from gym import logger
from gym.core import ObsType
from gym.error import (
from gym.vector.utils import (
from gym.vector.vector_env import VectorEnv
On deleting the object, checks that the vector environment is closed.