import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def test_overlapping_assignments():
    inds = _indices(ndims)
    for ind in inds:
        srcidx = tuple([a[0] for a in ind])
        dstidx = tuple([a[1] for a in ind])
        _check_assignment(srcidx, dstidx)