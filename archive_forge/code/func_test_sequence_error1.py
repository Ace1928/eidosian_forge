import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.compare import assertExpressionsEqual
def test_sequence_error1(self):
    try:
        sequence()
        self.fail('Expected ValueError')
    except ValueError:
        pass
    try:
        sequence(1, 2, 3, 4)
        self.fail('Expected ValueError')
    except ValueError:
        pass