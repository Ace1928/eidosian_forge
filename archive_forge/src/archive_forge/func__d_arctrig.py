import operator
import numpy as np
from . import markup
from .registry import unit_registry
from .decorators import memoize
def _d_arctrig(q1, out=None):
    if getattr(q1, 'dimensionality', None):
        raise ValueError('quantity must be dimensionless')
    return unit_registry['radian'].dimensionality