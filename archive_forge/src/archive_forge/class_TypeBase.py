from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename
class TypeBase:

    def __str__(self) -> str:
        raise NotImplementedError

    def declvar(self, x: str, init: Optional['Data']) -> str:
        if init is None:
            return f'{self} {x}'
        return f'{self} {x} = {init.code}'

    def assign(self, var: 'Data', value: 'Data') -> str:
        return f'{var.code} = {value.code}'