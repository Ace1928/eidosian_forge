import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
class _Ints(_Array):

    @property
    @abstractmethod
    def dtype(self) -> DTypesInt:
        ...

    @abstractmethod
    def fill(self, value: int) -> None:
        ...

    @abstractmethod
    def reshape(self, shape: Shape, *, order: str=...) -> '_Ints':
        ...