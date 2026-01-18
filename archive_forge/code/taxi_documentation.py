from contextlib import closing
from io import StringIO
from os import path
from typing import Optional
import numpy as np
from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled
Computes an action mask for the action space using the state information.