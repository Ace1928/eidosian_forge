import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@flatten_space.register(Discrete)
@flatten_space.register(MultiBinary)
@flatten_space.register(MultiDiscrete)
def _flatten_space_binary(space: Union[Discrete, MultiBinary, MultiDiscrete]) -> Box:
    return Box(low=0, high=1, shape=(flatdim(space),), dtype=space.dtype)