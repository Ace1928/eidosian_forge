import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@flatdim.register(Discrete)
def _flatdim_discrete(space: Discrete) -> int:
    return int(space.n)