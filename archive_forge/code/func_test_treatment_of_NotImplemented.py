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
def test_treatment_of_NotImplemented(self):
    a = masked_array([1.0, 2.0], mask=[1, 0])
    assert_raises(TypeError, operator.mul, a, 'abc')
    assert_raises(TypeError, operator.truediv, a, 'abc')

    class MyClass:
        __array_priority__ = a.__array_priority__ + 1

        def __mul__(self, other):
            return 'My mul'

        def __rmul__(self, other):
            return 'My rmul'
    me = MyClass()
    assert_(me * a == 'My mul')
    assert_(a * me == 'My rmul')

    class MyClass2:
        __array_priority__ = 100

        def __mul__(self, other):
            return 'Me2mul'

        def __rmul__(self, other):
            return 'Me2rmul'

        def __rdiv__(self, other):
            return 'Me2rdiv'
        __rtruediv__ = __rdiv__
    me_too = MyClass2()
    assert_(a.__mul__(me_too) is NotImplemented)
    assert_(all(multiply.outer(a, me_too) == 'Me2rmul'))
    assert_(a.__truediv__(me_too) is NotImplemented)
    assert_(me_too * a == 'Me2mul')
    assert_(a * me_too == 'Me2rmul')
    assert_(a / me_too == 'Me2rdiv')