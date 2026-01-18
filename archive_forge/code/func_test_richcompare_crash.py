import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_richcompare_crash(self):
    import operator as op

    class Foo:
        __array_priority__ = 1002

        def __array__(self, *args, **kwargs):
            raise Exception()
    rhs = Foo()
    lhs = np.array(1)
    for f in [op.lt, op.le, op.gt, op.ge]:
        assert_raises(TypeError, f, lhs, rhs)
    assert_(not op.eq(lhs, rhs))
    assert_(op.ne(lhs, rhs))