import builtins
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.expressions.expression import (
from cvxpy.tests.base_test import BaseTest
def test_some_np_ufunc_works(self) -> None:
    a = np.array([[1.0, 3.0], [3.0, 1.0]])
    b = np.int64(1)
    for ufunc in __BINARY_EXPRESSION_UFUNCS__:
        if ufunc is np.multiply:
            continue
        if ufunc is np.power:
            continue
        with pytest.raises(RuntimeError, match=__NUMPY_UFUNC_ERROR__):
            ufunc(self.x, a)
        with pytest.raises(RuntimeError, match=__NUMPY_UFUNC_ERROR__):
            ufunc(a, self.x, out=a)
        if ufunc is np.left_shift or ufunc is np.right_shift or ufunc is np.equal or (ufunc is np.less_equal) or (ufunc is np.greater_equal) or (ufunc is np.less) or (ufunc is np.greater):
            continue
        self.assertItemsAlmostEqual(ufunc(a, self.x).value, ufunc(a, self.x.value))
    for ufunc in __BINARY_EXPRESSION_UFUNCS__:
        if ufunc is np.matmul:
            continue
        if ufunc is np.power:
            continue
        with pytest.raises(RuntimeError, match=__NUMPY_UFUNC_ERROR__):
            ufunc(self.x, b)
        with pytest.raises(RuntimeError, match=__NUMPY_UFUNC_ERROR__):
            ufunc(b, self.x, out=b)
        if ufunc is np.left_shift or ufunc is np.right_shift or ufunc is np.equal or (ufunc is np.less_equal) or (ufunc is np.greater_equal) or (ufunc is np.less) or (ufunc is np.greater):
            continue
        self.assertItemsAlmostEqual(ufunc(b, self.x).value, ufunc(b, self.x.value))