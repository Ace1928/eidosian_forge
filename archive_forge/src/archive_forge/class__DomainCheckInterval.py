import builtins
import inspect
import operator
import warnings
import textwrap
import re
from functools import reduce
import numpy as np
import numpy.core.umath as umath
import numpy.core.numerictypes as ntypes
from numpy.core import multiarray as mu
from numpy import ndarray, amax, amin, iscomplexobj, bool_, _NoValue
from numpy import array as narray
from numpy.lib.function_base import angle
from numpy.compat import (
from numpy import expand_dims
from numpy.core.numeric import normalize_axis_tuple
frombuffer = _convert2ma(
fromfunction = _convert2ma(
class _DomainCheckInterval:
    """
    Define a valid interval, so that :

    ``domain_check_interval(a,b)(x) == True`` where
    ``x < a`` or ``x > b``.

    """

    def __init__(self, a, b):
        """domain_check_interval(a,b)(x) = true where x < a or y > b"""
        if a > b:
            a, b = (b, a)
        self.a = a
        self.b = b

    def __call__(self, x):
        """Execute the call behavior."""
        with np.errstate(invalid='ignore'):
            return umath.logical_or(umath.greater(x, self.b), umath.less(x, self.a))