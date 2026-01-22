import multiprocessing as mp
from collections import OrderedDict
from ctypes import c_bool
from functools import singledispatch
from typing import Union
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@read_from_shared_memory.register(Tuple)
def _read_tuple_from_shared_memory(space, shared_memory, n: int=1):
    return tuple((read_from_shared_memory(subspace, memory, n=n) for memory, subspace in zip(shared_memory, space.spaces)))