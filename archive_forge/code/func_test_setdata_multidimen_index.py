import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.base import IntegerSet
from pyomo.core.expr.numeric_expr import (
from pyomo.core.staleflag import StaleFlagManager
from pyomo.environ import (
from pyomo.core.base.units_container import units, pint_available, UnitsError
def test_setdata_multidimen_index(self):
    model = ConcreteModel()
    model.sindex = Set(initialize=[1])
    model.s = Set(model.sindex, dimen=2, initialize=[(1, 1), (1, 2), (1, 3)])
    model.x = Var(model.s[1], initialize=0, dense=True)
    self.assertEqual(len(model.x), 3)
    for i in model.s[1]:
        self.assertEqual(value(model.x[i]), 0)
    newIdx = (1, 4)
    self.assertFalse(newIdx in model.s[1])
    self.assertFalse(newIdx in model.x)
    model.s[1].add(newIdx)
    self.assertTrue(newIdx in model.s[1])
    self.assertFalse(newIdx in model.x)
    self.assertEqual(len(model.x), 3)
    for i in model.s[1]:
        self.assertEqual(value(model.x[i]), 0)
    self.assertEqual(len(model.x), 4)
    self.assertTrue(newIdx in model.s[1])
    self.assertTrue(newIdx in model.x)