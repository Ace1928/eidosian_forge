from __future__ import annotations
import math as _math
import typing as _typing
import warnings as _warnings
from operator import mul as _mul
from collections.abc import Iterable as _Iterable
from collections.abc import Iterator as _Iterator
class Vec3:
    __slots__ = ('x', 'y', 'z')
    'A three-dimensional vector represented as X Y Z coordinates.'

    def __init__(self, x: float=0.0, y: float=0.0, z: float=0.0) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __iter__(self) -> _Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    @_typing.overload
    def __getitem__(self, item: int) -> float:
        ...

    @_typing.overload
    def __getitem__(self, item: slice) -> tuple[float, ...]:
        ...

    def __getitem__(self, item):
        return (self.x, self.y, self.z)[item]

    def __setitem__(self, key, value):
        if type(key) is slice:
            for i, attr in enumerate(['x', 'y', 'z'][key]):
                setattr(self, attr, value[i])
        else:
            setattr(self, ['x', 'y', 'z'][key], value)

    def __len__(self) -> int:
        return 3

    @property
    def mag(self) -> float:
        """The magnitude, or length of the vector. The distance between the coordinates and the origin.

        Alias of abs(self).

        :type: float
        """
        return self.__abs__()

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> Vec3:
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __floordiv__(self, scalar: float) -> Vec3:
        return Vec3(self.x // scalar, self.y // scalar, self.z // scalar)

    def __abs__(self) -> float:
        return _math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)

    def __round__(self, ndigits: int | None=None) -> Vec3:
        return Vec3(*(round(v, ndigits) for v in self))

    def __radd__(self, other: Vec3 | int) -> Vec3:
        """Reverse add. Required for functionality with sum()"""
        if other == 0:
            return self
        else:
            return self.__add__(_typing.cast(Vec3, other))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Vec3) and self.x == other.x and (self.y == other.y) and (self.z == other.z)

    def __ne__(self, other: object) -> bool:
        return not isinstance(other, Vec3) or self.x != other.x or self.y != other.y or (self.z != other.z)

    def from_magnitude(self, magnitude: float) -> Vec3:
        """Create a new Vector of the given magnitude by normalizing,
        then scaling the vector. The rotation remains unchanged.
        """
        return self.normalize() * magnitude

    def limit(self, maximum: float) -> Vec3:
        """Limit the magnitude of the vector to the passed maximum value."""
        if self.x ** 2 + self.y ** 2 + self.z ** 2 > maximum * maximum * maximum:
            return self.from_magnitude(maximum)
        return self

    def cross(self, other: Vec3) -> Vec3:
        """Calculate the cross product of this vector and another 3D vector."""
        return Vec3(self.y * other.z - self.z * other.y, self.z * other.x - self.x * other.z, self.x * other.y - self.y * other.x)

    def dot(self, other: Vec3) -> float:
        """Calculate the dot product of this vector and another 3D vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def lerp(self, other: Vec3, alpha: float) -> Vec3:
        """Create a new Vec3 linearly interpolated between this vector and another Vec3.

        The `alpha` parameter dictates the amount of interpolation.
        This should be a value between 0.0 (this vector) and 1.0 (other vector).
        For example; 0.5 is the midway point between both vectors.
        """
        return Vec3(self.x + alpha * (other.x - self.x), self.y + alpha * (other.y - self.y), self.z + alpha * (other.z - self.z))

    def distance(self, other: Vec3) -> float:
        """Get the distance between this vector and another 3D vector."""
        return _math.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2 + (other.z - self.z) ** 2)

    def normalize(self) -> Vec3:
        """Normalize the vector to have a magnitude of 1. i.e. make it a unit vector."""
        try:
            d = self.__abs__()
            return Vec3(self.x / d, self.y / d, self.z / d)
        except ZeroDivisionError:
            return self

    def clamp(self, min_val: float, max_val: float) -> Vec3:
        """Restrict the value of the X, Y and Z components of the vector to be within the given values."""
        return Vec3(clamp(self.x, min_val, max_val), clamp(self.y, min_val, max_val), clamp(self.z, min_val, max_val))

    def __getattr__(self, attrs: str) -> Vec2 | Vec3 | Vec4:
        try:
            vec_class = {2: Vec2, 3: Vec3, 4: Vec4}[len(attrs)]
            return vec_class(*(self['xyz'.index(c)] for c in attrs))
        except Exception:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attrs}'") from None

    def __repr__(self) -> str:
        return f'Vec3({self.x}, {self.y}, {self.z})'