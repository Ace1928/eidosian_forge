from __future__ import annotations
import math as _math  # unexport, see #17
import typing
from functools import partial as _partial
from functools import wraps as _wraps  # unexport, see #17
def _dot_product(a: Triplet, b: Triplet) -> float:
    return sum((i * j for i, j in zip(a, b)))