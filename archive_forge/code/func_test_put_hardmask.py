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
def test_put_hardmask(self):
    d = arange(5)
    n = [0, 0, 0, 1, 1]
    m = make_mask(n)
    xh = array(d + 1, mask=m, hard_mask=True, copy=True)
    xh.put([4, 2, 0, 1, 3], [1, 2, 3, 4, 5])
    assert_equal(xh._data, [3, 4, 2, 4, 5])