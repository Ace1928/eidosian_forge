import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def verify_dual_values(self, places) -> None:
    for idx in range(len(self.constraints)):
        actual = self.constraints[idx].dual_value
        expect = self.expect_dual_vars[idx]
        if expect is not None:
            if isinstance(actual, list):
                for i in range(len(actual)):
                    act = actual[i]
                    exp = expect[i]
                    self.tester.assertItemsAlmostEqual(act, exp, places)
            else:
                self.tester.assertItemsAlmostEqual(actual, expect, places)