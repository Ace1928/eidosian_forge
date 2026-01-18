from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional
import numpy as np
from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled
def to_s(row, col):
    return row * ncol + col