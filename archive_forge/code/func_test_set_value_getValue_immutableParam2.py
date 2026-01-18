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
def test_set_value_getValue_immutableParam2(self):
    model = ConcreteModel()
    model.junk = Suffix()
    model.x = Var()
    model.X = Var([1, 2, 3])
    model.p = Param(initialize=1.0, mutable=False)
    model.P = Param([1, 2, 3], initialize=1.0, mutable=False)
    self.assertEqual(model.P.get_suffix_value('junk'), None)
    model.P.set_suffix_value('junk', 1.0, expand=False)
    self.assertEqual(model.P.get_suffix_value('junk'), 1.0)