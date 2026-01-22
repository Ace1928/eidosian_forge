import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@flatten.register(MultiDiscrete)
def _flatten_multidiscrete(space, x) -> np.ndarray:
    offsets = np.zeros((space.nvec.size + 1,), dtype=space.dtype)
    offsets[1:] = np.cumsum(space.nvec.flatten())
    onehot = np.zeros((offsets[-1],), dtype=space.dtype)
    onehot[offsets[:-1] + x.flatten()] = 1
    return onehot