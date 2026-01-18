import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, AbstractModel, SOSConstraint, Var, Set
def test_num_vars(self):
    self.M.x = Var([1, 2, 3])
    self.M.c = SOSConstraint(var=self.M.x, sos=1)
    self.assertEqual(self.M.c.num_variables(), 3)