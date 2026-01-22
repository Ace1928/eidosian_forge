from collections import OrderedDict
from functools import singledispatch
from typing import Iterable, Union
import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@create_empty_array.register(Box)
@create_empty_array.register(Discrete)
@create_empty_array.register(MultiDiscrete)
@create_empty_array.register(MultiBinary)
def _create_empty_array_base(space, n=1, fn=np.zeros):
    shape = space.shape if n is None else (n,) + space.shape
    return fn(shape, dtype=space.dtype)