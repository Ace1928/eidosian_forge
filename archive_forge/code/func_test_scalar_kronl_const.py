from typing import Tuple
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def test_scalar_kronl_const(self):
    self.scalar_kronl(param=False)