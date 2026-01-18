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
def test_set_record_slice(self):
    base = self.data['base']
    base_a, base_b, base_c = (base['a'], base['b'], base['c'])
    base[:3] = (pi, pi, 'pi')
    assert_equal(base_a.dtype, int)
    assert_equal(base_a._data, [3, 3, 3, 4, 5])
    assert_equal(base_b.dtype, float)
    assert_equal(base_b._data, [pi, pi, pi, 4.4, 5.5])
    assert_equal(base_c.dtype, '|S8')
    assert_equal(base_c._data, [b'pi', b'pi', b'pi', b'four', b'five'])