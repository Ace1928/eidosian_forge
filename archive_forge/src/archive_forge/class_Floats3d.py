import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
class Floats3d(_Array3d, _Floats):
    """3-dimensional array of floats"""
    T: 'Floats3d'

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield (lambda v: validate_array(v, ndim=3, dtype='f'))

    @abstractmethod
    def __iter__(self) -> Iterator[Floats2d]:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _3_KeyScalar) -> float:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _3_Key1d) -> Floats1d:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _3_Key2d) -> Floats2d:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _3_Key3d) -> 'Floats3d':
        ...

    @abstractmethod
    def __getitem__(self, key: _3_AllKeys) -> _F3_AllReturns:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _3_KeyScalar, value: float) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _3_Key1d, value: Floats1d) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _3_Key2d, value: Floats2d) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _3_Key3d, value: 'Floats3d') -> None:
        ...

    @abstractmethod
    def __setitem__(self, key: _3_AllKeys, value: _F3_AllReturns) -> None:
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Tru, axis: _3_AllAx=None, out: Optional['Floats3d']=None) -> 'Floats3d':
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal, axis: OneAx, out: Optional[Floats2d]=None) -> Floats2d:
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal, axis: TwoAx, out: Optional[Floats1d]=None) -> Floats1d:
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal, axis: Optional[ThreeAx], out=None) -> float:
        ...

    @abstractmethod
    def sum(self, *, keepdims: bool=False, axis: _3_AllAx=None, out: Union[None, Floats1d, Floats2d, 'Floats3d']=None) -> _3F_ReduceResults:
        ...