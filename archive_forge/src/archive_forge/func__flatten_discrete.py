import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@flatten.register(Discrete)
def _flatten_discrete(space, x) -> np.ndarray:
    onehot = np.zeros(space.n, dtype=space.dtype)
    onehot[x - space.start] = 1
    return onehot