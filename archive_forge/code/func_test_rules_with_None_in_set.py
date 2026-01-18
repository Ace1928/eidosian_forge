import re
import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
from pyomo.scripting.pyomo_main import main
from pyomo.core import (
from pyomo.common.tee import capture_output
from io import StringIO
def test_rules_with_None_in_set(self):

    def noarg_rule(b):
        b.args = ()

    def onearg_rule(b, i):
        b.args = (i,)

    def twoarg_rule(b, i, j):
        b.args = (i, j)
    m = ConcreteModel()
    m.b1 = Block(rule=noarg_rule)
    self.assertEqual(m.b1.args, ())
    m.b2 = Block([None], rule=onearg_rule)
    self.assertEqual(m.b2[None].args, (None,))
    m.b3 = Block([(None, 1)], rule=twoarg_rule)
    self.assertEqual(m.b3[None, 1].args, (None, 1))