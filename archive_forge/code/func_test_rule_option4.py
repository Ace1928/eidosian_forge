import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_rule_option4(self):
    """Test rule option"""
    model = self.create_model()
    model.y = Var(initialize=2)
    model.c = ObjectiveList(rule=((i + 1) * model.y for i in range(3)))
    self.assertEqual(len(model.c), 3)
    self.assertEqual(model.c[1](), 2)