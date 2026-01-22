import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
def _graph_unflatten(space, x):
    ret = None
    if space is not None and x is not None:
        if isinstance(space, Box):
            ret = x.reshape(-1, *space.shape)
        elif isinstance(space, Discrete):
            ret = np.asarray(np.nonzero(x))[-1, :]
    return ret