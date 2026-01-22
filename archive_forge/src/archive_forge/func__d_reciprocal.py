import operator
import numpy as np
from . import markup
from .registry import unit_registry
from .decorators import memoize
def _d_reciprocal(q1, out=None):
    return q1._dimensionality ** (-1)