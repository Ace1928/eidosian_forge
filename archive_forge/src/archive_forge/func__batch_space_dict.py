from collections import OrderedDict
from copy import deepcopy
from functools import singledispatch
from typing import Iterator
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@batch_space.register(Dict)
def _batch_space_dict(space, n=1):
    return Dict(OrderedDict([(key, batch_space(subspace, n=n)) for key, subspace in space.spaces.items()]), seed=deepcopy(space.np_random))