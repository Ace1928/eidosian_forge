import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
@dataclass
class Pairs(Generic[_P]):
    """Dataclass for pairs of sequences that allows indexing into the sequences
    while keeping them aligned.
    """
    one: _P
    two: _P

    def __getitem__(self, index) -> 'Pairs[_P]':
        return Pairs(self.one[index], self.two[index])

    def __len__(self) -> int:
        return len(self.one)