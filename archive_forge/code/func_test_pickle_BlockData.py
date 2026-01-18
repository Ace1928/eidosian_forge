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
def test_pickle_BlockData(self):
    model = ConcreteModel()
    model.b = Block([1, 2, 3])
    model.junk = Suffix()
    self.assertEqual(model.junk.get(model.b[1]), None)
    model.junk.set_value(model.b[1], 1.0)
    self.assertEqual(model.junk.get(model.b[1]), 1.0)
    inst = pickle.loads(pickle.dumps(model))
    self.assertEqual(inst.junk.get(model.b[1]), None)
    self.assertEqual(inst.junk.get(inst.b[1]), 1.0)