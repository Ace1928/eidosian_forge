from __future__ import annotations
import math as _math
import typing as _typing
import warnings as _warnings
from operator import mul as _mul
from collections.abc import Iterable as _Iterable
from collections.abc import Iterator as _Iterator
def to_mat4(self) -> Mat4:
    w = self.w
    x = self.x
    y = self.y
    z = self.z
    a = 1 - (y ** 2 + z ** 2) * 2
    b = 2 * (x * y - z * w)
    c = 2 * (x * z + y * w)
    e = 2 * (x * y + z * w)
    f = 1 - (x ** 2 + z ** 2) * 2
    g = 2 * (y * z - x * w)
    i = 2 * (x * z - y * w)
    j = 2 * (y * z + x * w)
    k = 1 - (x ** 2 + y ** 2) * 2
    return Mat4((a, b, c, 0.0, e, f, g, 0.0, i, j, k, 0.0, 0.0, 0.0, 0.0, 1.0))