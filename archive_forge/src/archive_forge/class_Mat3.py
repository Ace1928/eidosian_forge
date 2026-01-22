from __future__ import annotations
import math as _math
import typing as _typing
import warnings as _warnings
from operator import mul as _mul
from collections.abc import Iterable as _Iterable
from collections.abc import Iterator as _Iterator
class Mat3(tuple):
    """A 3x3 Matrix

    `Mat3` is an immutable 3x3 Matrix, including most common
    operators.

    A Matrix can be created with a list or tuple of 12 values.
    If no values are provided, an "identity matrix" will be created
    (1.0 on the main diagonal). Mat3 objects are immutable, so
    all operations return a new Mat3 object.

    .. note:: Matrix multiplication is performed using the "@" operator.
    """

    def __new__(cls: type[Mat3T], values: _Iterable[float]=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)) -> Mat3T:
        new = super().__new__(cls, values)
        assert len(new) == 9, 'A 3x3 Matrix requires 9 values'
        return new

    def scale(self, sx: float, sy: float) -> Mat3:
        return self @ Mat3((1.0 / sx, 0.0, 0.0, 0.0, 1.0 / sy, 0.0, 0.0, 0.0, 1.0))

    def translate(self, tx: float, ty: float) -> Mat3:
        return self @ Mat3((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -tx, ty, 1.0))

    def rotate(self, phi: float) -> Mat3:
        s = _math.sin(_math.radians(phi))
        c = _math.cos(_math.radians(phi))
        return self @ Mat3((c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0))

    def shear(self, sx: float, sy: float) -> Mat3:
        return self @ Mat3((1.0, sy, 0.0, sx, 1.0, 0.0, 0.0, 0.0, 1.0))

    def __add__(self, other: Mat3) -> Mat3:
        if not isinstance(other, Mat3):
            raise TypeError('Can only add to other Mat3 types')
        return Mat3((s + o for s, o in zip(self, other)))

    def __sub__(self, other: Mat3) -> Mat3:
        if not isinstance(other, Mat3):
            raise TypeError('Can only subtract from other Mat3 types')
        return Mat3((s - o for s, o in zip(self, other)))

    def __pos__(self) -> Mat3:
        return self

    def __neg__(self) -> Mat3:
        return Mat3((-v for v in self))

    def __round__(self, ndigits: int | None=None) -> Mat3:
        return Mat3((round(v, ndigits) for v in self))

    def __mul__(self, other: object) -> _typing.NoReturn:
        raise NotImplementedError('Please use the @ operator for Matrix multiplication.')

    @_typing.overload
    def __matmul__(self, other: Vec3) -> Vec3:
        ...

    @_typing.overload
    def __matmul__(self, other: Mat3) -> Mat3:
        ...

    def __matmul__(self, other):
        if isinstance(other, Vec3):
            r0 = self[0::3]
            r1 = self[1::3]
            r2 = self[2::3]
            return Vec3(sum(map(_mul, r0, other)), sum(map(_mul, r1, other)), sum(map(_mul, r2, other)))
        if not isinstance(other, Mat3):
            raise TypeError('Can only multiply with Mat3 or Vec3 types')
        r0 = self[0::3]
        r1 = self[1::3]
        r2 = self[2::3]
        c0 = other[0:3]
        c1 = other[3:6]
        c2 = other[6:9]
        return Mat3((sum(map(_mul, c0, r0)), sum(map(_mul, c0, r1)), sum(map(_mul, c0, r2)), sum(map(_mul, c1, r0)), sum(map(_mul, c1, r1)), sum(map(_mul, c1, r2)), sum(map(_mul, c2, r0)), sum(map(_mul, c2, r1)), sum(map(_mul, c2, r2))))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self[0:3]}\n    {self[3:6]}\n    {self[6:9]}'