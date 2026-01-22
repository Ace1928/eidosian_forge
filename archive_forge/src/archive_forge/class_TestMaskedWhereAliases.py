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
class TestMaskedWhereAliases:

    def test_masked_values(self):
        res = masked_values(np.array([-32768.0]), np.int16(-32768))
        assert_equal(res.mask, [True])
        res = masked_values(np.inf, np.inf)
        assert_equal(res.mask, True)
        res = np.ma.masked_values(np.inf, -np.inf)
        assert_equal(res.mask, False)
        res = np.ma.masked_values([1, 2, 3, 4], 5, shrink=True)
        assert_(res.mask is np.ma.nomask)
        res = np.ma.masked_values([1, 2, 3, 4], 5, shrink=False)
        assert_equal(res.mask, [False] * 4)