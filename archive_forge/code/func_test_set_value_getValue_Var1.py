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
def test_set_value_getValue_Var1(self):
    model = ConcreteModel()
    model.junk = Suffix()
    model.x = Var()
    model.X = Var([1, 2, 3], dense=True)
    model.junk.set_value(model.X, 1.0)
    model.junk.set_value(model.X[1], 2.0)
    self.assertEqual(model.junk.get(model.X), None)
    self.assertEqual(model.junk.get(model.X[1]), 2.0)
    self.assertEqual(model.junk.get(model.X[2]), 1.0)
    self.assertEqual(model.junk.get(model.x), None)
    model.junk.set_value(model.x, 3.0)
    model.junk.set_value(model.X[2], 3.0)
    self.assertEqual(model.junk.get(model.X), None)
    self.assertEqual(model.junk.get(model.X[1]), 2.0)
    self.assertEqual(model.junk.get(model.X[2]), 3.0)
    self.assertEqual(model.junk.get(model.x), 3.0)
    model.junk.set_value(model.X, 1.0, expand=False)
    self.assertEqual(model.junk.get(model.X), 1.0)