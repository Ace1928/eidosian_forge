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
def test_on_ndarray(self):
    a = np.array([1, 2, 3, 4])
    m = array(a, mask=False)
    test = anom(a)
    assert_equal(test, m.anom())
    test = reshape(a, (2, 2))
    assert_equal(test, m.reshape(2, 2))