import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
def test_testUfuncRegression(self):
    for f in ['sqrt', 'log', 'log10', 'exp', 'conjugate', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh', 'absolute', 'fabs', 'negative', 'floor', 'ceil', 'logical_not', 'add', 'subtract', 'multiply', 'divide', 'true_divide', 'floor_divide', 'remainder', 'fmod', 'hypot', 'arctan2', 'equal', 'not_equal', 'less_equal', 'greater_equal', 'less', 'greater', 'logical_and', 'logical_or', 'logical_xor']:
        try:
            uf = getattr(umath, f)
        except AttributeError:
            uf = getattr(fromnumeric, f)
        mf = getattr(numpy.ma.core, f)
        args = self.d[:uf.nin]
        ur = uf(*args)
        mr = mf(*args)
        assert_equal(ur.filled(0), mr.filled(0), f)
        assert_mask_equal(ur.mask, mr.mask, err_msg=f)