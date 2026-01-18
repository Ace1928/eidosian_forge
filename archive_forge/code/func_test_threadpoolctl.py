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
def test_threadpoolctl():
    try:
        import threadpoolctl
        if not hasattr(threadpoolctl, 'register'):
            pytest.skip('threadpoolctl too old')
            return
    except ImportError:
        pytest.skip('no threadpoolctl')
        return
    with threadpoolctl.threadpool_limits(limits=4):
        assert_equal(fmm.PARALLELISM, 4)
    with threadpoolctl.threadpool_limits(limits=2, user_api='scipy'):
        assert_equal(fmm.PARALLELISM, 2)