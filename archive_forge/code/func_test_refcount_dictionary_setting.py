import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
def test_refcount_dictionary_setting(self):
    names = ['name1']
    formats = ['f8']
    titles = ['t1']
    offsets = [0]
    d = dict(names=names, formats=formats, titles=titles, offsets=offsets)
    refcounts = {k: sys.getrefcount(i) for k, i in d.items()}
    np.dtype(d)
    refcounts_new = {k: sys.getrefcount(i) for k, i in d.items()}
    assert refcounts == refcounts_new