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
@suppress_copy_mask_on_assignment
def test_varstd_specialcases(self):
    nout = np.array(-1, dtype=float)
    mout = array(-1, dtype=float)
    x = array(arange(10), mask=True)
    for methodname in ('var', 'std'):
        method = getattr(x, methodname)
        assert_(method() is masked)
        assert_(method(0) is masked)
        assert_(method(-1) is masked)
        method(out=mout)
        assert_(mout is not masked)
        assert_equal(mout.mask, True)
        method(out=nout)
        assert_(np.isnan(nout))
    x = array(arange(10), mask=True)
    x[-1] = 9
    for methodname in ('var', 'std'):
        method = getattr(x, methodname)
        assert_(method(ddof=1) is masked)
        assert_(method(0, ddof=1) is masked)
        assert_(method(-1, ddof=1) is masked)
        method(out=mout, ddof=1)
        assert_(mout is not masked)
        assert_equal(mout.mask, True)
        method(out=nout, ddof=1)
        assert_(np.isnan(nout))