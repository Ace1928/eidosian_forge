import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_tuple_subclass(self):
    arr = np.ones((5, 5))

    class TupleSubclass(tuple):
        pass
    index = ([1], [1])
    index = TupleSubclass(index)
    assert_(arr[index].shape == (1,))
    assert_(arr[index,].shape != (1,))