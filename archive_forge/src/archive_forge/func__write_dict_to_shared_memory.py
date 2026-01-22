import multiprocessing as mp
from collections import OrderedDict
from ctypes import c_bool
from functools import singledispatch
from typing import Union
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@write_to_shared_memory.register(Dict)
def _write_dict_to_shared_memory(space, index, values, shared_memory):
    for key, subspace in space.spaces.items():
        write_to_shared_memory(subspace, index, values[key], shared_memory[key])