import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_path_type_input_invalid(self):
    path_test = self.build_operands('ab,bc,cd,de->ae')
    exp_path = ['einsum_path', (2, 3), (0, 1)]
    assert_raises(RuntimeError, np.einsum, *path_test, optimize=exp_path)
    assert_raises(RuntimeError, np.einsum_path, *path_test, optimize=exp_path)
    path_test = self.build_operands('a,a,a->a')
    exp_path = ['einsum_path', (1,), (0, 1)]
    assert_raises(RuntimeError, np.einsum, *path_test, optimize=exp_path)
    assert_raises(RuntimeError, np.einsum_path, *path_test, optimize=exp_path)