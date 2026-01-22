from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename
class Scalar(TypeBase):

    def __init__(self, dtype: npt.DTypeLike) -> None:
        self.dtype = numpy.dtype(dtype)

    def __str__(self) -> str:
        dtype = self.dtype
        if dtype == numpy.float16:
            dtype = numpy.dtype('float32')
        return get_typename(dtype)

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, TypeBase)
        return isinstance(other, Scalar) and self.dtype == other.dtype

    def __hash__(self) -> int:
        return hash(self.dtype)