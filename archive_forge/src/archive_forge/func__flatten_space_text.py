import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@flatten_space.register(Text)
def _flatten_space_text(space: Text) -> Box:
    return Box(low=0, high=len(space.character_set), shape=(space.max_length,), dtype=np.int32)