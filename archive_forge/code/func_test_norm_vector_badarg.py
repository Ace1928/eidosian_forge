import warnings
import numpy as np
from numpy import linalg, arange, float64, array, dot, transpose
from numpy.testing import (
def test_norm_vector_badarg(self):
    assert_raises(ValueError, linalg.norm, array([1.0, 2.0, 3.0]), 'fro')