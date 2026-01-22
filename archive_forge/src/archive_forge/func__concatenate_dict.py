from collections import OrderedDict
from functools import singledispatch
from typing import Iterable, Union
import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@concatenate.register(Dict)
def _concatenate_dict(space, items, out):
    return OrderedDict([(key, concatenate(subspace, [item[key] for item in items], out[key])) for key, subspace in space.spaces.items()])