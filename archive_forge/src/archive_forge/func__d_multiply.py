import operator
import numpy as np
from . import markup
from .registry import unit_registry
from .decorators import memoize
def _d_multiply(q1, q2, out=None):
    try:
        return q1._dimensionality * q2._dimensionality
    except AttributeError:
        try:
            return q1.dimensionality
        except:
            return q2.dimensionality