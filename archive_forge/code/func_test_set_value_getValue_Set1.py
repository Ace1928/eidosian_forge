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
def test_set_value_getValue_Set1(self):
    model = ConcreteModel()
    model.junk = Suffix()
    model.x = Var()
    model.X = Var([1, 2, 3])
    model.s = Set(initialize=[1, 2, 3])
    model.S = Set([1, 2, 3], initialize={1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]})
    model.junk.set_value(model.S, 1.0)
    model.junk.set_value(model.S[1], 2.0)
    self.assertEqual(model.junk.get(model.S), None)
    self.assertEqual(model.junk.get(model.S[1]), 2.0)
    self.assertEqual(model.junk.get(model.S[2]), 1.0)
    self.assertEqual(model.junk.get(model.s), None)
    model.junk.set_value(model.s, 3.0)
    model.junk.set_value(model.S[2], 3.0)
    self.assertEqual(model.junk.get(model.S), None)
    self.assertEqual(model.junk.get(model.S[1]), 2.0)
    self.assertEqual(model.junk.get(model.S[2]), 3.0)
    self.assertEqual(model.junk.get(model.s), 3.0)
    model.junk.set_value(model.S, 1.0, expand=False)
    self.assertEqual(model.junk.get(model.S), 1.0)