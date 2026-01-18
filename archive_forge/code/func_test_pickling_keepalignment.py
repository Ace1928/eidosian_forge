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
def test_pickling_keepalignment(self):
    a = arange(10)
    a.shape = (-1, 2)
    b = a.T
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        test = pickle.loads(pickle.dumps(b, protocol=proto))
        assert_equal(test, b)