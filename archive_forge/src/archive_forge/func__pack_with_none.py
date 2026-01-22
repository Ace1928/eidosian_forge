import torch
from collections import OrderedDict
import weakref
import warnings
from typing import Any, Tuple
def _pack_with_none(self, indices, values, size):
    res = [None] * size
    for idx, val in zip(indices, values):
        res[idx] = val
    return tuple(res)