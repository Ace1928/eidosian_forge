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
def test_set_value_getValue_Objective1(self):
    model = ConcreteModel()
    model.junk = Suffix()
    model.x = Var()
    model.X = Var([1, 2, 3])
    model.obj = Objective(expr=sum_product(model.X) + model.x)
    model.OBJ = Objective([1, 2, 3], rule=lambda model, i: model.X[i])
    model.junk.set_value(model.OBJ, 1.0)
    model.junk.set_value(model.OBJ[1], 2.0)
    self.assertEqual(model.junk.get(model.OBJ), None)
    self.assertEqual(model.junk.get(model.OBJ[1]), 2.0)
    self.assertEqual(model.junk.get(model.OBJ[2]), 1.0)
    self.assertEqual(model.junk.get(model.obj), None)
    model.junk.set_value(model.obj, 3.0)
    model.junk.set_value(model.OBJ[2], 3.0)
    self.assertEqual(model.junk.get(model.OBJ), None)
    self.assertEqual(model.junk.get(model.OBJ[1]), 2.0)
    self.assertEqual(model.junk.get(model.OBJ[2]), 3.0)
    self.assertEqual(model.junk.get(model.obj), 3.0)
    model.junk.set_value(model.OBJ, 1.0, expand=False)
    self.assertEqual(model.junk.get(model.OBJ), 1.0)