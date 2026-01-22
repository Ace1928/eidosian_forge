import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
from pyomo.environ import (
def Xtest_list_domain_bad_duplicates(self):
    with self.assertRaises(ValueError) as cm:
        self.model = ConcreteModel()
        self.model.y = Var([1, 2], within=[1.7, 2, 3])