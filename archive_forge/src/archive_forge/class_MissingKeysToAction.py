from collections import deque
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import gym.error
from gym import Env, logger
from gym.core import ActType, ObsType
from gym.error import DependencyNotInstalled
from gym.logger import deprecation
class MissingKeysToAction(Exception):
    """Raised when the environment does not have a default ``keys_to_action`` mapping."""