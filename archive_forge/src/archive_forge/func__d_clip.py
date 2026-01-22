import operator
import numpy as np
from . import markup
from .registry import unit_registry
from .decorators import memoize
def _d_clip(a1, a2, a3, q):
    return q.dimensionality