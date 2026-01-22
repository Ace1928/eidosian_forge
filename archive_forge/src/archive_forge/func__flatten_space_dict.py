import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@flatten_space.register(Dict)
def _flatten_space_dict(space: Dict) -> Union[Box, Dict]:
    if space.is_np_flattenable:
        space_list = [flatten_space(s) for s in space.spaces.values()]
        return Box(low=np.concatenate([s.low for s in space_list]), high=np.concatenate([s.high for s in space_list]), dtype=np.result_type(*[s.dtype for s in space_list]))
    return Dict(spaces=OrderedDict(((key, flatten_space(space)) for key, space in space.spaces.items())))