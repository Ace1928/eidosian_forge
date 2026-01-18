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
def test_update_values(self):
    model = ConcreteModel()
    model.junk = Suffix()
    model.x = Var()
    model.y = Var()
    model.z = Var([1, 2])
    model.junk.set_value(model.x, 0.0)
    self.assertEqual(model.junk.get(model.x), 0.0)
    self.assertEqual(model.junk.get(model.y), None)
    self.assertEqual(model.junk.get(model.z), None)
    self.assertEqual(model.junk.get(model.z[1]), None)
    self.assertEqual(model.junk.get(model.z[2]), None)
    model.junk.update_values([(model.x, 1.0), (model.y, 2.0), (model.z, 3.0)])
    self.assertEqual(model.junk.get(model.x), 1.0)
    self.assertEqual(model.junk.get(model.y), 2.0)
    self.assertEqual(model.junk.get(model.z), None)
    self.assertEqual(model.junk.get(model.z[1]), 3.0)
    self.assertEqual(model.junk.get(model.z[2]), 3.0)
    model.junk.clear()
    model.junk.update_values([(model.x, 1.0), (model.y, 2.0), (model.z, 3.0)], expand=False)
    self.assertEqual(model.junk.get(model.x), 1.0)
    self.assertEqual(model.junk.get(model.y), 2.0)
    self.assertEqual(model.junk.get(model.z), 3.0)
    self.assertEqual(model.junk.get(model.z[1]), None)
    self.assertEqual(model.junk.get(model.z[2]), None)