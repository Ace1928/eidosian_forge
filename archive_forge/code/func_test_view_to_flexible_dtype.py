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
def test_view_to_flexible_dtype(self):
    data, a, controlmask = self.data
    test = a.view([('A', float), ('B', float)])
    assert_equal(test.mask.dtype.names, ('A', 'B'))
    assert_equal(test['A'], a['a'])
    assert_equal(test['B'], a['b'])
    test = a[0].view([('A', float), ('B', float)])
    assert_(isinstance(test, MaskedArray))
    assert_equal(test.mask.dtype.names, ('A', 'B'))
    assert_equal(test['A'], a['a'][0])
    assert_equal(test['B'], a['b'][0])
    test = a[-1].view([('A', float), ('B', float)])
    assert_(isinstance(test, MaskedArray))
    assert_equal(test.dtype.names, ('A', 'B'))
    assert_equal(test['A'], a['a'][-1])
    assert_equal(test['B'], a['b'][-1])