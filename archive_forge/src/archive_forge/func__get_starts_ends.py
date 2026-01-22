import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
def _get_starts_ends(self) -> Ints1d:
    if self.starts_ends is None:
        xp = get_array_module(self.lengths)
        self.starts_ends = xp.empty(self.lengths.size + 1, dtype='i')
        self.starts_ends[0] = 0
        self.lengths.cumsum(out=self.starts_ends[1:])
    return self.starts_ends