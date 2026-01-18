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
def test_tolist_specialcase(self):
    a = array([(0, 1), (2, 3)], dtype=[('a', int), ('b', int)])
    for entry in a:
        for item in entry.tolist():
            assert_(not isinstance(item, np.generic))
    a.mask[0] = (0, 1)
    for entry in a:
        for item in entry.tolist():
            assert_(not isinstance(item, np.generic))