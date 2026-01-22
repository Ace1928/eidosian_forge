from collections import OrderedDict
from copy import deepcopy
from functools import singledispatch
from typing import Iterator
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@iterate.register(Discrete)
def _iterate_discrete(space, items):
    raise TypeError('Unable to iterate over a space of type `Discrete`.')