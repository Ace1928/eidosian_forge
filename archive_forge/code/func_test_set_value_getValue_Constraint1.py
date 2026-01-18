import os
import itertools
import logging
import pickle
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.suffix import (
from pyomo.environ import (
from io import StringIO
def test_set_value_getValue_Constraint1(self):
    model = ConcreteModel()
    model.junk = Suffix()
    model.x = Var()
    model.X = Var([1, 2, 3])
    model.c = Constraint(expr=model.x >= 1)
    model.C = Constraint([1, 2, 3], rule=lambda model, i: model.X[i] >= 1)
    model.junk.set_value(model.C, 1.0)
    model.junk.set_value(model.C[1], 2.0)
    self.assertEqual(model.junk.get(model.C), None)
    self.assertEqual(model.junk.get(model.C[1]), 2.0)
    self.assertEqual(model.junk.get(model.C[2]), 1.0)
    self.assertEqual(model.junk.get(model.c), None)
    model.junk.set_value(model.c, 3.0)
    model.junk.set_value(model.C[2], 3.0)
    self.assertEqual(model.junk.get(model.C), None)
    self.assertEqual(model.junk.get(model.C[1]), 2.0)
    self.assertEqual(model.junk.get(model.C[2]), 3.0)
    self.assertEqual(model.junk.get(model.c), 3.0)
    model.junk.set_value(model.C, 1.0, expand=False)
    self.assertEqual(model.junk.get(model.C), 1.0)