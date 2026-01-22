import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def _indices_for_axis():
    """Returns (src, dst) pairs of indices."""
    res = []
    for nelems in (0, 2, 3):
        ind = _indices_for_nelems(nelems)
        res.extend(itertools.product(ind, ind))
    return res