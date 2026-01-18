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
def test_set_value_getValue_Var3(self):
    model = ConcreteModel()
    model.junk = Suffix()
    model.x = Var()
    model.X = Var([1, 2, 3], dense=True)
    model.X.set_suffix_value(model.junk, 1.0)
    model.X[1].set_suffix_value(model.junk, 2.0)
    self.assertEqual(model.X.get_suffix_value(model.junk), None)
    self.assertEqual(model.X[1].get_suffix_value(model.junk), 2.0)
    self.assertEqual(model.X[2].get_suffix_value(model.junk), 1.0)
    self.assertEqual(model.x.get_suffix_value(model.junk), None)
    model.x.set_suffix_value(model.junk, 3.0)
    model.X[2].set_suffix_value(model.junk, 3.0)
    self.assertEqual(model.X.get_suffix_value(model.junk), None)
    self.assertEqual(model.X[1].get_suffix_value(model.junk), 2.0)
    self.assertEqual(model.X[2].get_suffix_value(model.junk), 3.0)
    self.assertEqual(model.x.get_suffix_value(model.junk), 3.0)
    model.X.set_suffix_value(model.junk, 1.0, expand=False)
    self.assertEqual(model.X.get_suffix_value(model.junk), 1.0)