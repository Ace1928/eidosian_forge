from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy

    Tries to identify __radd__ and friends (so the METH_COEXIST flag can be applied).

    There's no great consequence if it inadvertently identifies a few other methods
    so just use a simple rule rather than an exact list.
    