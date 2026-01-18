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
def test_varmean_nomask(self):
    foo = array([1, 2, 3, 4], dtype='f8')
    bar = array([1, 2, 3, 4], dtype='f8')
    assert_equal(type(foo.mean()), np.float64)
    assert_equal(type(foo.var()), np.float64)
    assert (foo.mean() == bar.mean()) is np.bool_(True)
    foo = array(np.arange(16).reshape((4, 4)), dtype='f8')
    bar = empty(4, dtype='f4')
    assert_equal(type(foo.mean(axis=1)), MaskedArray)
    assert_equal(type(foo.var(axis=1)), MaskedArray)
    assert_(foo.mean(axis=1, out=bar) is bar)
    assert_(foo.var(axis=1, out=bar) is bar)