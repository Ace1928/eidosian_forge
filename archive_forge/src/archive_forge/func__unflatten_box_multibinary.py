import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@unflatten.register(Box)
@unflatten.register(MultiBinary)
def _unflatten_box_multibinary(space: Union[Box, MultiBinary], x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=space.dtype).reshape(space.shape)