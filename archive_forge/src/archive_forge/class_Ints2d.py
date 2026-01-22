import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
class Ints2d(_Array2d, _Ints):
    """2-dimensional array of ints."""
    T: 'Ints2d'

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield (lambda v: validate_array(v, ndim=2, dtype='i'))

    @abstractmethod
    def __iter__(self) -> Iterator[Ints1d]:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _2_KeyScalar) -> int:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _2_Key1d) -> Ints1d:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _2_Key2d) -> 'Ints2d':
        ...

    @abstractmethod
    def __getitem__(self, key: _2_AllKeys) -> _I2_AllReturns:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _2_KeyScalar, value: int) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _2_Key1d, value: Ints1d) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _2_Key2d, value: 'Ints2d') -> None:
        ...

    @abstractmethod
    def __setitem__(self, key: _2_AllKeys, value: _I2_AllReturns) -> None:
        ...

    @overload
    @abstractmethod
    def sum(self, keepdims: Fal=False, axis: int=-1, out: Optional['Ints1d']=None) -> Ints1d:
        ...

    @overload
    @abstractmethod
    def sum(self, keepdims: Tru, axis: int=-1, out: Optional['Ints2d']=None) -> 'Ints2d':
        ...

    @abstractmethod
    def sum(self, keepdims: bool=False, axis: int=-1, out: Optional[Union['Ints1d', 'Ints2d']]=None) -> Union['Ints2d', Ints1d]:
        ...