import json
import os
from os.path import abspath, dirname, join
import pickle
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml_available
from pyomo.common.tempfiles import TempfileManager
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.opt import check_available_solvers
from pyomo.opt.parallel.local import SolverManager_Serial
def test_stats1(self):
    model = ConcreteModel()
    model.x = Var([1, 2])

    def obj_rule(model, i):
        return sum_product(model.x)
    model.obj = Objective([1, 2], rule=obj_rule)

    def c_rule(model, i):
        expr = 0
        for j in [1, 2]:
            expr += j * model.x[j]
        return expr == 0
    model.c = Constraint([1, 2], rule=c_rule)
    self.assertEqual(model.nvariables(), 2)
    self.assertEqual(model.nobjectives(), 2)
    self.assertEqual(model.nconstraints(), 2)