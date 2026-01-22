import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@flatten.register(Text)
def _flatten_text(space: Text, x: str) -> np.ndarray:
    arr = np.full(shape=(space.max_length,), fill_value=len(space.character_set), dtype=np.int32)
    for i, val in enumerate(x):
        arr[i] = space.character_index(val)
    return arr