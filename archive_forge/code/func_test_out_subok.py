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
def test_out_subok(self):
    for subok in (True, False):
        a = np.array(0.5)
        o = np.empty(())
        r = np.add(a, 2, o, subok=subok)
        assert_(r is o)
        r = np.add(a, 2, out=o, subok=subok)
        assert_(r is o)
        r = np.add(a, 2, out=(o,), subok=subok)
        assert_(r is o)
        d = np.array(5.7)
        o1 = np.empty(())
        o2 = np.empty((), dtype=np.int32)
        r1, r2 = np.frexp(d, o1, None, subok=subok)
        assert_(r1 is o1)
        r1, r2 = np.frexp(d, None, o2, subok=subok)
        assert_(r2 is o2)
        r1, r2 = np.frexp(d, o1, o2, subok=subok)
        assert_(r1 is o1)
        assert_(r2 is o2)
        r1, r2 = np.frexp(d, out=(o1, None), subok=subok)
        assert_(r1 is o1)
        r1, r2 = np.frexp(d, out=(None, o2), subok=subok)
        assert_(r2 is o2)
        r1, r2 = np.frexp(d, out=(o1, o2), subok=subok)
        assert_(r1 is o1)
        assert_(r2 is o2)
        with assert_raises(TypeError):
            r1, r2 = np.frexp(d, out=o1, subok=subok)
        assert_raises(TypeError, np.add, a, 2, o, o, subok=subok)
        assert_raises(TypeError, np.add, a, 2, o, out=o, subok=subok)
        assert_raises(TypeError, np.add, a, 2, None, out=o, subok=subok)
        assert_raises(ValueError, np.add, a, 2, out=(o, o), subok=subok)
        assert_raises(ValueError, np.add, a, 2, out=(), subok=subok)
        assert_raises(TypeError, np.add, a, 2, [], subok=subok)
        assert_raises(TypeError, np.add, a, 2, out=[], subok=subok)
        assert_raises(TypeError, np.add, a, 2, out=([],), subok=subok)
        o.flags.writeable = False
        assert_raises(ValueError, np.add, a, 2, o, subok=subok)
        assert_raises(ValueError, np.add, a, 2, out=o, subok=subok)
        assert_raises(ValueError, np.add, a, 2, out=(o,), subok=subok)