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
@unittest.skipIf('glpk' not in solvers, 'glpk solver is not available')
def test_store_to_skip_stale_vars(self):
    model = ConcreteModel()
    model.A = RangeSet(1, 4)
    model.x = Var(model.A, bounds=(-1, 1))

    def obj_rule(model):
        return sum_product(model.x)
    model.obj = Objective(rule=obj_rule)

    def c_rule(model):
        expr = 0
        for i in model.A:
            expr += i * model.x[i]
        return expr == 0
    model.c = Constraint(rule=c_rule)
    opt = SolverFactory('glpk')
    results = opt.solve(model, symbolic_solver_labels=True)
    model.x[1].fix()
    results = opt.solve(model, symbolic_solver_labels=True)
    model.solutions.store_to(results, skip_stale_vars=False)
    for index in model.A:
        self.assertIn(model.x[index].getname(), results.solution.variable.keys())
    model.solutions.store_to(results, skip_stale_vars=True)
    for index in model.A:
        if index == 1:
            self.assertNotIn(model.x[index].getname(), results.solution.variable.keys())
        else:
            self.assertIn(model.x[index].getname(), results.solution.variable.keys())