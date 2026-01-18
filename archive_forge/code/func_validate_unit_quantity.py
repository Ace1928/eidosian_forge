import copy
from functools import wraps
import numpy as np
from . import markup
from .dimensionality import Dimensionality, p_dict
from .registry import unit_registry
from .decorators import with_doc
def validate_unit_quantity(value):
    try:
        assert isinstance(value, Quantity)
        assert value.shape in ((), (1,))
        assert value.magnitude == 1
    except AssertionError:
        raise ValueError('units must be a scalar Quantity with unit magnitude, got %s' % value)
    return value