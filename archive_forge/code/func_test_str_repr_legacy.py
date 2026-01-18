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
def test_str_repr_legacy(self):
    oldopts = np.get_printoptions()
    np.set_printoptions(legacy='1.13')
    try:
        a = array([0, 1, 2], mask=[False, True, False])
        assert_equal(str(a), '[0 -- 2]')
        assert_equal(repr(a), 'masked_array(data = [0 -- 2],\n             mask = [False  True False],\n       fill_value = 999999)\n')
        a = np.ma.arange(2000)
        a[1:50] = np.ma.masked
        assert_equal(repr(a), 'masked_array(data = [0 -- -- ..., 1997 1998 1999],\n             mask = [False  True  True ..., False False False],\n       fill_value = 999999)\n')
    finally:
        np.set_printoptions(**oldopts)