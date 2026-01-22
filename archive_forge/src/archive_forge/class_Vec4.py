from __future__ import annotations
import math as _math
import typing as _typing
import warnings as _warnings
from operator import mul as _mul
from collections.abc import Iterable as _Iterable
from collections.abc import Iterator as _Iterator
class Vec4:
    __slots__ = ('x', 'y', 'z', 'w')
    'A four-dimensional vector represented as X Y Z W coordinates.'

    def __init__(self, x: float=0.0, y: float=0.0, z: float=0.0, w: float=0.0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __iter__(self) -> _Iterator[float]:
        yield self.x
        yield self.y
        yield self.z
        yield self.w

    @_typing.overload
    def __getitem__(self, item: int) -> float:
        ...

    @_typing.overload
    def __getitem__(self, item: slice) -> tuple[float, ...]:
        ...

    def __getitem__(self, item):
        return (self.x, self.y, self.z, self.w)[item]

    def __setitem__(self, key, value):
        if type(key) is slice:
            for i, attr in enumerate(['x', 'y', 'z', 'w'][key]):
                setattr(self, attr, value[i])
        else:
            setattr(self, ['x', 'y', 'z', 'w'][key], value)

    def __len__(self) -> int:
        return 4

    def __add__(self, other: Vec4) -> Vec4:
        return Vec4(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)

    def __sub__(self, other: Vec4) -> Vec4:
        return Vec4(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)

    def __mul__(self, scalar: float) -> Vec4:
        return Vec4(self.x * scalar, self.y * scalar, self.z * scalar, self.w * scalar)

    def __truediv__(self, scalar: float) -> Vec4:
        return Vec4(self.x / scalar, self.y / scalar, self.z / scalar, self.w / scalar)

    def __floordiv__(self, scalar: float) -> Vec4:
        return Vec4(self.x // scalar, self.y // scalar, self.z // scalar, self.w // scalar)

    def __abs__(self) -> float:
        return _math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2 + self.w ** 2)

    def __neg__(self) -> Vec4:
        return Vec4(-self.x, -self.y, -self.z, -self.w)

    def __round__(self, ndigits: int | None=None) -> Vec4:
        return Vec4(*(round(v, ndigits) for v in self))

    def __radd__(self, other: Vec4 | int) -> Vec4:
        if other == 0:
            return self
        else:
            return self.__add__(_typing.cast(Vec4, other))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Vec4) and self.x == other.x and (self.y == other.y) and (self.z == other.z) and (self.w == other.w)

    def __ne__(self, other: object) -> bool:
        return not isinstance(other, Vec4) or self.x != other.x or self.y != other.y or (self.z != other.z) or (self.w != other.w)

    def lerp(self, other: Vec4, alpha: float) -> Vec4:
        """Create a new Vec4 linearly interpolated between this one and another Vec4.

        The `alpha` parameter dictates the amount of interpolation.
        This should be a value between 0.0 (this vector) and 1.0 (other vector).
        For example; 0.5 is the midway point between both vectors.
        """
        return Vec4(self.x + alpha * (other.x - self.x), self.y + alpha * (other.y - self.y), self.z + alpha * (other.z - self.z), self.w + alpha * (other.w - self.w))

    def distance(self, other: Vec4) -> float:
        return _math.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2 + (other.z - self.z) ** 2 + (other.w - self.w) ** 2)

    def normalize(self) -> Vec4:
        """Normalize the vector to have a magnitude of 1. i.e. make it a unit vector."""
        d = self.__abs__()
        if d:
            return Vec4(self.x / d, self.y / d, self.z / d, self.w / d)
        return self

    def clamp(self, min_val: float, max_val: float) -> Vec4:
        return Vec4(clamp(self.x, min_val, max_val), clamp(self.y, min_val, max_val), clamp(self.z, min_val, max_val), clamp(self.w, min_val, max_val))

    def dot(self, other: Vec4) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    def __getattr__(self, attrs: str) -> Vec2 | Vec3 | Vec4:
        try:
            vec_class = {2: Vec2, 3: Vec3, 4: Vec4}[len(attrs)]
            return vec_class(*(self['xyzw'.index(c)] for c in attrs))
        except Exception:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attrs}'") from None

    def __repr__(self) -> str:
        return f'Vec4({self.x}, {self.y}, {self.z}, {self.w})'