import copy
from functools import wraps
import numpy as np
from . import markup
from .dimensionality import Dimensionality, p_dict
from .registry import unit_registry
from .decorators import with_doc

        Return a tuple for pickling a Quantity.
        