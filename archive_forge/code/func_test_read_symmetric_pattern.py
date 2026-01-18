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
def test_read_symmetric_pattern(self):
    a = [[1, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 0, 1, 1]]
    self.check_read(_symmetric_pattern_example, a, (5, 5, 7, 'coordinate', 'pattern', 'symmetric'))