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
def test_read_over64bit_integer_dense(self):
    self.check_read(_over64bit_integer_dense_example, None, (2, 2, 4, 'array', 'integer', 'general'), dense=True, over32=True, over64=True)