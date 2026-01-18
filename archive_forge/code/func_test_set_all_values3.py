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
def test_set_all_values3(self):
    model = ConcreteModel()
    model.junk = Suffix()
    model.x = Var()
    model.y = Var([1, 2, 3], dense=True)
    model.z = Var([1, 2, 3], dense=True)
    model.y[2].set_suffix_value(model.junk, 1.0)
    model.z.set_suffix_value(model.junk, 2.0)
    self.assertTrue(model.x.get_suffix_value(model.junk) is None)
    self.assertTrue(model.y.get_suffix_value(model.junk) is None)
    self.assertTrue(model.y[1].get_suffix_value(model.junk) is None)
    self.assertEqual(model.y[2].get_suffix_value(model.junk), 1.0)
    self.assertEqual(model.z.get_suffix_value(model.junk), None)
    self.assertEqual(model.z[1].get_suffix_value(model.junk), 2.0)
    model.junk.set_all_values(3.0)
    self.assertTrue(model.x.get_suffix_value(model.junk) is None)
    self.assertTrue(model.y.get_suffix_value(model.junk) is None)
    self.assertTrue(model.y[1].get_suffix_value(model.junk) is None)
    self.assertEqual(model.y[2].get_suffix_value(model.junk), 3.0)
    self.assertEqual(model.z.get_suffix_value(model.junk), None)
    self.assertEqual(model.z[1].get_suffix_value(model.junk), 3.0)