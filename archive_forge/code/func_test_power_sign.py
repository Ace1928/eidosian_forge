import numpy as np
import scipy.sparse as sp
import cvxpy
from cvxpy.tests.base_test import BaseTest
def test_power_sign(self) -> None:
    x = cvxpy.Variable(pos=True)
    self.assertTrue((x ** 1).is_nonneg())
    self.assertFalse((x ** 1).is_nonpos())