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
def test_set_value_getValue_Objective3(self):
    model = ConcreteModel()
    model.junk = Suffix()
    model.x = Var()
    model.X = Var([1, 2, 3])
    model.obj = Objective(expr=sum_product(model.X) + model.x)
    model.OBJ = Objective([1, 2, 3], rule=lambda model, i: model.X[i])
    model.OBJ.set_suffix_value(model.junk, 1.0)
    model.OBJ[1].set_suffix_value(model.junk, 2.0)
    self.assertEqual(model.OBJ.get_suffix_value(model.junk), None)
    self.assertEqual(model.OBJ[1].get_suffix_value(model.junk), 2.0)
    self.assertEqual(model.OBJ[2].get_suffix_value(model.junk), 1.0)
    self.assertEqual(model.obj.get_suffix_value(model.junk), None)
    model.obj.set_suffix_value(model.junk, 3.0)
    model.OBJ[2].set_suffix_value(model.junk, 3.0)
    self.assertEqual(model.OBJ.get_suffix_value(model.junk), None)
    self.assertEqual(model.OBJ[1].get_suffix_value(model.junk), 2.0)
    self.assertEqual(model.OBJ[2].get_suffix_value(model.junk), 3.0)
    self.assertEqual(model.obj.get_suffix_value(model.junk), 3.0)
    model.OBJ.set_suffix_value(model.junk, 1.0, expand=False)
    self.assertEqual(model.OBJ.get_suffix_value(model.junk), 1.0)