import multiprocessing as mp
from collections import OrderedDict
from ctypes import c_bool
from functools import singledispatch
from typing import Union
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@create_shared_memory.register(Tuple)
def _create_tuple_shared_memory(space, n: int=1, ctx=mp):
    return tuple((create_shared_memory(subspace, n=n, ctx=ctx) for subspace in space.spaces))