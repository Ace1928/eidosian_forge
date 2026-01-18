import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_recarray_fromarrays(self):
    x1 = np.array([1, 2, 3, 4])
    x2 = np.array(['a', 'dd', 'xyz', '12'])
    x3 = np.array([1.1, 2, 3, 4])
    r = np.rec.fromarrays([x1, x2, x3], names='a,b,c')
    assert_equal(r[1].item(), (2, 'dd', 2.0))
    x1[1] = 34
    assert_equal(r.a, np.array([1, 2, 3, 4]))