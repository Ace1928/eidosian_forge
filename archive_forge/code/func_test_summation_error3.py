import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.compare import assertExpressionsEqual
def test_summation_error3(self):
    model = AbstractModel()
    model.A = Set(initialize=[1, 2, 3])
    model.B = Param(model.A, initialize={1: 100, 2: 200, 3: 300}, mutable=True)
    model.x = Var(model.A)
    instance = model.create_instance()
    try:
        expr = sum_product(denom=(instance.x, instance.B))
        self.fail('Expected ValueError')
    except ValueError:
        pass