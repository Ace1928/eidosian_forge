import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
@property
def _supported_types(self) -> Tuple[Type, ...]:
    cls = list(self._NUMBER_TYPES)
    if NUMPY_AVAILABLE:
        cls.append(np.number)
    return tuple(cls)