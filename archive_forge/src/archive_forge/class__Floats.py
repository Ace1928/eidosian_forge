import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
class _Floats(_Array):

    @property
    @abstractmethod
    def dtype(self) -> DTypesFloat:
        ...

    @abstractmethod
    def fill(self, value: float) -> None:
        ...

    @abstractmethod
    def reshape(self, shape: Shape, *, order: str=...) -> '_Floats':
        ...