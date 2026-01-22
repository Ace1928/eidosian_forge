import functools
import re
import warnings
from .._utils import set_module
import numpy.core.numeric as NX
from numpy.core import (isscalar, abs, finfo, atleast_1d, hstack, dot, array,
from numpy.core import overrides
from numpy.lib.twodim_base import diag, vander
from numpy.lib.function_base import trim_zeros
from numpy.lib.type_check import iscomplex, real, imag, mintypecode
from numpy.linalg import eigvals, lstsq, inv
def _roots_dispatcher(p):
    return p