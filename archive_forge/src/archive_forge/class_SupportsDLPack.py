from __future__ import annotations
from cupy.cuda import Device
import sys
from typing import (
from ._array_object import Array
from numpy import (
class SupportsDLPack(Protocol):

    def __dlpack__(self, /, *, stream: None=...) -> PyCapsule:
        ...