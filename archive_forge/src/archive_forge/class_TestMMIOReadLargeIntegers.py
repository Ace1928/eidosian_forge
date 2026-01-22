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
class TestMMIOReadLargeIntegers:

    def setup_method(self):
        self.tmpdir = mkdtemp()
        self.fn = os.path.join(self.tmpdir, 'testfile.mtx')

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def check_read(self, example, a, info, dense, over32, over64):
        with open(self.fn, 'w') as f:
            f.write(example)
        assert_equal(mminfo(self.fn), info)
        if over32 and np.intp(0).itemsize < 8 and (mmwrite == scipy.io._mmio.mmwrite) or over64:
            assert_raises(OverflowError, mmread, self.fn)
        else:
            b = mmread(self.fn)
            if not dense:
                b = b.toarray()
            assert_equal(a, b)

    def test_read_32bit_integer_dense(self):
        a = array([[2 ** 31 - 1, 2 ** 31 - 1], [2 ** 31 - 2, 2 ** 31 - 2]], dtype=np.int64)
        self.check_read(_32bit_integer_dense_example, a, (2, 2, 4, 'array', 'integer', 'general'), dense=True, over32=False, over64=False)

    def test_read_32bit_integer_sparse(self):
        a = array([[2 ** 31 - 1, 0], [0, 2 ** 31 - 2]], dtype=np.int64)
        self.check_read(_32bit_integer_sparse_example, a, (2, 2, 2, 'coordinate', 'integer', 'symmetric'), dense=False, over32=False, over64=False)

    def test_read_64bit_integer_dense(self):
        a = array([[2 ** 31, -2 ** 31], [-2 ** 63 + 2, 2 ** 63 - 1]], dtype=np.int64)
        self.check_read(_64bit_integer_dense_example, a, (2, 2, 4, 'array', 'integer', 'general'), dense=True, over32=True, over64=False)

    def test_read_64bit_integer_sparse_general(self):
        a = array([[2 ** 31, 2 ** 63 - 1], [0, 2 ** 63 - 1]], dtype=np.int64)
        self.check_read(_64bit_integer_sparse_general_example, a, (2, 2, 3, 'coordinate', 'integer', 'general'), dense=False, over32=True, over64=False)

    def test_read_64bit_integer_sparse_symmetric(self):
        a = array([[2 ** 31, -2 ** 63 + 1], [-2 ** 63 + 1, 2 ** 63 - 1]], dtype=np.int64)
        self.check_read(_64bit_integer_sparse_symmetric_example, a, (2, 2, 3, 'coordinate', 'integer', 'symmetric'), dense=False, over32=True, over64=False)

    def test_read_64bit_integer_sparse_skew(self):
        a = array([[2 ** 31, -2 ** 63 + 1], [2 ** 63 - 1, 2 ** 63 - 1]], dtype=np.int64)
        self.check_read(_64bit_integer_sparse_skew_example, a, (2, 2, 3, 'coordinate', 'integer', 'skew-symmetric'), dense=False, over32=True, over64=False)

    def test_read_over64bit_integer_dense(self):
        self.check_read(_over64bit_integer_dense_example, None, (2, 2, 4, 'array', 'integer', 'general'), dense=True, over32=True, over64=True)

    def test_read_over64bit_integer_sparse(self):
        self.check_read(_over64bit_integer_sparse_example, None, (2, 2, 2, 'coordinate', 'integer', 'symmetric'), dense=False, over32=True, over64=True)