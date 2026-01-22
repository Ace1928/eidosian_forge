import operator
import numpy as np
from . import markup
from .registry import unit_registry
from .decorators import memoize
def _d_copy(q1, out=None):
    return q1.dimensionality