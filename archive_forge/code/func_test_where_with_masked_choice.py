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
def test_where_with_masked_choice(self):
    x = arange(10)
    x[3] = masked
    c = x >= 8
    z = where(c, x, masked)
    assert_(z.dtype is x.dtype)
    assert_(z[3] is masked)
    assert_(z[4] is masked)
    assert_(z[7] is masked)
    assert_(z[8] is not masked)
    assert_(z[9] is not masked)
    assert_equal(x, z)
    z = where(c, masked, x)
    assert_(z.dtype is x.dtype)
    assert_(z[3] is masked)
    assert_(z[4] is not masked)
    assert_(z[7] is not masked)
    assert_(z[8] is masked)
    assert_(z[9] is masked)