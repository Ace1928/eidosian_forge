from io import StringIO
import os
import sys
import types
import json
from copy import deepcopy
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.block import (
import pyomo.core.expr as EXPR
from pyomo.opt import check_available_solvers
from pyomo.gdp import Disjunct
def test_pseudomap_len(self):
    m = Block()
    m.a = Constraint()
    m.b = Constraint()
    m.c = Constraint()
    m.z = Objective()
    m.x = Objective()
    m.v = Objective()
    m.y = Objective()
    m.w = Objective()
    m.b.deactivate()
    m.z.deactivate()
    m.w.deactivate()
    self.assertEqual(len(m.component_map()), 8)
    self.assertEqual(len(m.component_map(active=True)), 5)
    self.assertEqual(len(m.component_map(active=False)), 3)
    self.assertEqual(len(m.component_map(Constraint)), 3)
    self.assertEqual(len(m.component_map(Constraint, active=True)), 2)
    self.assertEqual(len(m.component_map(Constraint, active=False)), 1)
    self.assertEqual(len(m.component_map(Objective)), 5)
    self.assertEqual(len(m.component_map(Objective, active=True)), 3)
    self.assertEqual(len(m.component_map(Objective, active=False)), 2)