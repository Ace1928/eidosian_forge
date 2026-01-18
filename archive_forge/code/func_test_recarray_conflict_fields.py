import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_recarray_conflict_fields(self):
    ra = np.rec.array([(1, 'abc', 2.3), (2, 'xyz', 4.2), (3, 'wrs', 1.3)], names='field, shape, mean')
    ra.mean = [1.1, 2.2, 3.3]
    assert_array_almost_equal(ra['mean'], [1.1, 2.2, 3.3])
    assert_(type(ra.mean) is type(ra.var))
    ra.shape = (1, 3)
    assert_(ra.shape == (1, 3))
    ra.shape = ['A', 'B', 'C']
    assert_array_equal(ra['shape'], [['A', 'B', 'C']])
    ra.field = 5
    assert_array_equal(ra['field'], [[5, 5, 5]])
    assert_(isinstance(ra.field, collections.abc.Callable))