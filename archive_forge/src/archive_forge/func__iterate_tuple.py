from collections import OrderedDict
from copy import deepcopy
from functools import singledispatch
from typing import Iterator
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@iterate.register(Tuple)
def _iterate_tuple(space, items):
    if all((isinstance(subspace, Space) and (not isinstance(subspace, BaseGymSpaces + (Tuple, Dict))) for subspace in space.spaces)):
        return iter(items)
    return zip(*[iterate(subspace, items[i]) for i, subspace in enumerate(space.spaces)])