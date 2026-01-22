from __future__ import annotations
import math as _math  # unexport, see #17
import typing
from functools import partial as _partial
from functools import wraps as _wraps  # unexport, see #17
def _max_chroma_for_lh(l: float, h: float) -> float:
    hrad = _math.radians(h)
    lengths = [_length_of_ray_until_intersect(hrad, bound) for bound in _get_bounds(l)]
    return min((length for length in lengths if length >= 0))