from __future__ import annotations
import math as _math  # unexport, see #17
import typing
from functools import partial as _partial
from functools import wraps as _wraps  # unexport, see #17
def _max_safe_chroma_for_l(l: float) -> float:
    return min((_distance_line_from_origin(bound) for bound in _get_bounds(l)))