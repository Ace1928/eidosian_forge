import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@flatten_space.register(Tuple)
def _flatten_space_tuple(space: Tuple) -> Union[Box, Tuple]:
    if space.is_np_flattenable:
        space_list = [flatten_space(s) for s in space.spaces]
        return Box(low=np.concatenate([s.low for s in space_list]), high=np.concatenate([s.high for s in space_list]), dtype=np.result_type(*[s.dtype for s in space_list]))
    return Tuple(spaces=[flatten_space(s) for s in space.spaces])