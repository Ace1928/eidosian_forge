import copy
from functools import wraps
import numpy as np
from . import markup
from .dimensionality import Dimensionality, p_dict
from .registry import unit_registry
from .decorators import with_doc
def _reconstruct_quantity(subtype, baseclass, baseshape, basetype):
    """Internal function that builds a new MaskedArray from the
    information stored in a pickle.

    """
    _data = np.ndarray.__new__(baseclass, baseshape, basetype)
    return subtype.__new__(subtype, _data, dtype=basetype)