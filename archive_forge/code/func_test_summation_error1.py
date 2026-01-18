import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.compare import assertExpressionsEqual
def test_summation_error1(self):
    try:
        sum_product()
        self.fail('Expected ValueError')
    except ValueError:
        pass