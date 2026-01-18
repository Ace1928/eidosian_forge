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
def test_oddfeatures_2(self):
    x = array([1.0, 2.0, 3.0, 4.0, 5.0])
    c = array([1, 1, 1, 0, 0])
    x[2] = masked
    z = where(c, x, -x)
    assert_equal(z, [1.0, 2.0, 0.0, -4.0, -5])
    c[0] = masked
    z = where(c, x, -x)
    assert_equal(z, [1.0, 2.0, 0.0, -4.0, -5])
    assert_(z[0] is masked)
    assert_(z[1] is not masked)
    assert_(z[2] is masked)