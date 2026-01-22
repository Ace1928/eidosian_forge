from collections import OrderedDict
from copy import deepcopy
from functools import singledispatch
from typing import Iterator
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@iterate.register(Dict)
def _iterate_dict(space, items):
    keys, values = zip(*[(key, iterate(subspace, items[key])) for key, subspace in space.spaces.items()])
    for item in zip(*values):
        yield OrderedDict([(key, value) for key, value in zip(keys, item)])