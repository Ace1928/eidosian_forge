import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@flatdim.register(MultiDiscrete)
def _flatdim_multidiscrete(space: MultiDiscrete) -> int:
    return int(np.sum(space.nvec))