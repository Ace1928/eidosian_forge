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
def test_set_all_values1(self):
    model = ConcreteModel()
    model.junk = Suffix()
    model.x = Var()
    model.y = Var([1, 2, 3], dense=True)
    model.z = Var([1, 2, 3], dense=True)
    model.junk.set_value(model.y[2], 1.0)
    model.junk.set_value(model.z, 2.0)
    self.assertTrue(model.junk.get(model.x) is None)
    self.assertTrue(model.junk.get(model.y) is None)
    self.assertTrue(model.junk.get(model.y[1]) is None)
    self.assertEqual(model.junk.get(model.y[2]), 1.0)
    self.assertEqual(model.junk.get(model.z), None)
    self.assertEqual(model.junk.get(model.z[1]), 2.0)
    model.junk.set_all_values(3.0)
    self.assertTrue(model.junk.get(model.x) is None)
    self.assertTrue(model.junk.get(model.y) is None)
    self.assertTrue(model.junk.get(model.y[1]) is None)
    self.assertEqual(model.junk.get(model.y[2]), 3.0)
    self.assertEqual(model.junk.get(model.z), None)
    self.assertEqual(model.junk.get(model.z[1]), 3.0)