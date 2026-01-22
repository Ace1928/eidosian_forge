import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
from pyomo.environ import (
def Xtest_set_domain_empty(self):
    with self.assertRaises(ValueError) as cm:
        self.model = ConcreteModel()
        self.model.y = Var([2, 2], within=set([]))