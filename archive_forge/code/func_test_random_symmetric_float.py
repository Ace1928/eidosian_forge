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
def test_random_symmetric_float(self):
    sz = (20, 20)
    a = np.random.random(sz)
    a = a + transpose(a)
    a = scipy.sparse.csr_matrix(a)
    self.check(a, (20, 20, 210, 'coordinate', 'real', 'symmetric'))