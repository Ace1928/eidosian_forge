from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional
import numpy as np
from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled
def update_probability_matrix(row, col, action):
    newrow, newcol = inc(row, col, action)
    newstate = to_s(newrow, newcol)
    newletter = desc[newrow, newcol]
    terminated = bytes(newletter) in b'GH'
    reward = float(newletter == b'G')
    return (newstate, reward, terminated)