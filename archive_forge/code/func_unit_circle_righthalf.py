import copy
from functools import lru_cache
from weakref import WeakValueDictionary
import numpy as np
import matplotlib as mpl
from . import _api, _path
from .cbook import _to_unmasked_float_array, simple_linear_interpolation
from .bezier import BezierSegment
@classmethod
def unit_circle_righthalf(cls):
    """
        Return a `Path` of the right half of a unit circle.

        See `Path.circle` for the reference on the approximation used.
        """
    if cls._unit_circle_righthalf is None:
        MAGIC = 0.2652031
        SQRTHALF = np.sqrt(0.5)
        MAGIC45 = SQRTHALF * MAGIC
        vertices = np.array([[0.0, -1.0], [MAGIC, -1.0], [SQRTHALF - MAGIC45, -SQRTHALF - MAGIC45], [SQRTHALF, -SQRTHALF], [SQRTHALF + MAGIC45, -SQRTHALF + MAGIC45], [1.0, -MAGIC], [1.0, 0.0], [1.0, MAGIC], [SQRTHALF + MAGIC45, SQRTHALF - MAGIC45], [SQRTHALF, SQRTHALF], [SQRTHALF - MAGIC45, SQRTHALF + MAGIC45], [MAGIC, 1.0], [0.0, 1.0], [0.0, -1.0]], float)
        codes = np.full(14, cls.CURVE4, dtype=cls.code_type)
        codes[0] = cls.MOVETO
        codes[-1] = cls.CLOSEPOLY
        cls._unit_circle_righthalf = cls(vertices, codes, readonly=True)
    return cls._unit_circle_righthalf