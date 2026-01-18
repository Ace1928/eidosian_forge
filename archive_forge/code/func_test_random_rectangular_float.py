from tempfile import mkdtemp
import os
import io
import shutil
import textwrap
import numpy as np
from numpy import array, transpose, pi
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
import scipy.sparse
import scipy.io._mmio
import scipy.io._fast_matrix_market as fmm
def test_random_rectangular_float(self):
    sz = (20, 15)
    a = np.random.random(sz)
    a = scipy.sparse.csr_matrix(a)
    self.check(a, (20, 15, 300, 'coordinate', 'real', 'general'))