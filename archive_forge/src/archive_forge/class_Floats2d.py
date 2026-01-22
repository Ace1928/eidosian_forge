import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
class Floats2d(_Array2d, _Floats):
    """2-dimensional array of floats"""
    T: 'Floats2d'

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield (lambda v: validate_array(v, ndim=2, dtype='f'))

    @abstractmethod
    def __iter__(self) -> Iterator[Floats1d]:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _2_KeyScalar) -> float:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _2_Key1d) -> Floats1d:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _2_Key2d) -> 'Floats2d':
        ...

    @abstractmethod
    def __getitem__(self, key: _2_AllKeys) -> _F2_AllReturns:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _2_KeyScalar, value: float) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _2_Key1d, value: Union[float, Floats1d]) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _2_Key2d, value: _F2_AllReturns) -> None:
        ...

    @abstractmethod
    def __setitem__(self, key: _2_AllKeys, value: _F2_AllReturns) -> None:
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Tru, axis: _2_AllAx=None, out: Optional['Floats2d']=None) -> 'Floats2d':
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal=False, axis: OneAx, out: Optional[Floats1d]=None) -> Floats1d:
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal=False, axis: TwoAx, out=None) -> float:
        ...

    @abstractmethod
    def sum(self, *, keepdims: bool=False, axis: _2_AllAx=None, out: Union[None, 'Floats1d', 'Floats2d']=None) -> _2F_ReduceResults:
        ...