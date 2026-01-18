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
def test_set_value_getValue_mutableParam1(self):
    model = ConcreteModel()
    model.junk = Suffix()
    model.x = Var()
    model.X = Var([1, 2, 3])
    model.p = Param(initialize=1.0, mutable=True)
    model.P = Param([1, 2, 3], initialize=1.0, mutable=True)
    model.junk.set_value(model.P, 1.0)
    model.junk.set_value(model.P[1], 2.0)
    self.assertEqual(model.junk.get(model.P), None)
    self.assertEqual(model.junk.get(model.P[1]), 2.0)
    self.assertEqual(model.junk.get(model.P[2]), 1.0)
    self.assertEqual(model.junk.get(model.p), None)
    model.junk.set_value(model.p, 3.0)
    model.junk.set_value(model.P[2], 3.0)
    self.assertEqual(model.junk.get(model.P), None)
    self.assertEqual(model.junk.get(model.P[1]), 2.0)
    self.assertEqual(model.junk.get(model.P[2]), 3.0)
    self.assertEqual(model.junk.get(model.p), 3.0)
    model.junk.set_value(model.P, 1.0, expand=False)
    self.assertEqual(model.junk.get(model.P), 1.0)