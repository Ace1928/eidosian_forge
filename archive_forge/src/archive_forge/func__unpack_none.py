import torch
from collections import OrderedDict
import weakref
import warnings
from typing import Any, Tuple
def _unpack_none(self, indices, values):
    res = []
    for idx in indices:
        res.append(values[idx])
    return tuple(res)