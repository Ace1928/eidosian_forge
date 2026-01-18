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
def test_pickling_maskedconstant(self):
    mc = np.ma.masked
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        mc_pickled = pickle.loads(pickle.dumps(mc, protocol=proto))
        assert_equal(mc_pickled._baseclass, mc._baseclass)
        assert_equal(mc_pickled._mask, mc._mask)
        assert_equal(mc_pickled._data, mc._data)