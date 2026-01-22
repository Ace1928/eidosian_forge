from collections import OrderedDict
from copy import deepcopy
from functools import singledispatch
from typing import Iterator
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@batch_space.register(MultiBinary)
def _batch_space_multibinary(space, n=1):
    return Box(low=0, high=1, shape=(n,) + space.shape, dtype=space.dtype, seed=deepcopy(space.np_random))