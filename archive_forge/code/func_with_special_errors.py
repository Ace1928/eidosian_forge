import os
import functools
import operator
from scipy._lib import _pep440
import numpy as np
from numpy.testing import assert_
import pytest
import scipy.special as sc
def with_special_errors(func):
    """
    Enable special function errors (such as underflow, overflow,
    loss of precision, etc.)
    """

    @functools.wraps(func)
    def wrapper(*a, **kw):
        with sc.errstate(all='raise'):
            res = func(*a, **kw)
        return res
    return wrapper