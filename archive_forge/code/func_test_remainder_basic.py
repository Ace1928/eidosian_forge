import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
def test_remainder_basic(self):
    dt = np.typecodes['AllInteger'] + np.typecodes['Float']
    for op in [floor_divide_and_remainder, np.divmod]:
        for dt1, dt2 in itertools.product(dt, dt):
            for sg1, sg2 in itertools.product(_signs(dt1), _signs(dt2)):
                fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                a = np.array(sg1 * 71, dtype=dt1)
                b = np.array(sg2 * 19, dtype=dt2)
                div, rem = op(a, b)
                assert_equal(div * b + rem, a, err_msg=msg)
                if sg2 == -1:
                    assert_(b < rem <= 0, msg)
                else:
                    assert_(b > rem >= 0, msg)