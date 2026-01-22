import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
from pyomo.environ import (
def Xtest_pyomo_Set_dat_file_domain_empty(self):
    with self.assertRaises(ValueError) as cm:
        self.model = AbstractModel()
        self.model.s = Set()
        self.model.y = Var([1, 2], within=self.model.s)
        self.instance = self.model.create_instance(currdir + 'vars_dat_file_empty.dat')