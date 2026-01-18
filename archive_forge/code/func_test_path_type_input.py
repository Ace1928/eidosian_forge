import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_path_type_input(self):
    path_test = self.build_operands('dcc,fce,ea,dbf->ab')
    path, path_str = np.einsum_path(*path_test, optimize=False)
    self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])
    path, path_str = np.einsum_path(*path_test, optimize=True)
    self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 1), (0, 1)])
    exp_path = ['einsum_path', (0, 2), (0, 2), (0, 1)]
    path, path_str = np.einsum_path(*path_test, optimize=exp_path)
    self.assert_path_equal(path, exp_path)
    noopt = np.einsum(*path_test, optimize=False)
    opt = np.einsum(*path_test, optimize=exp_path)
    assert_almost_equal(noopt, opt)