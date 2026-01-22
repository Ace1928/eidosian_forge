import multiprocessing as mp
from collections import OrderedDict
from ctypes import c_bool
from functools import singledispatch
from typing import Union
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@write_to_shared_memory.register(Tuple)
def _write_tuple_to_shared_memory(space, index, values, shared_memory):
    for value, memory, subspace in zip(values, shared_memory, space.spaces):
        write_to_shared_memory(subspace, index, value, memory)