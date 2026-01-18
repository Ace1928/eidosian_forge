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
@unittest.skipIf(not yaml_available, 'YAML not available available')
def test_solve_with_store5(self):
    model = ConcreteModel()
    model.A = RangeSet(1, 4)
    model.b = Block()
    model.b.x = Var(model.A, bounds=(-1, 1))
    model.b.obj = Objective(expr=sum_product(model.b.x))
    model.c = Constraint(expr=model.b.x[1] >= 0)
    smanager = SolverManager_Serial()
    ah = smanager.queue(model, solver='glpk', load_solutions=False)
    results = smanager.wait_for(ah)
    self.assertEqual(len(model.solutions), 0)
    self.assertEqual(len(results.solution), 1)
    model.solutions.load_from(results)
    self.assertEqual(len(model.solutions), 1)
    self.assertEqual(len(results.solution), 1)
    model.solutions.store_to(results)
    results.write(filename=join(currdir, 'solve_with_store8.out'), format='json')
    with open(join(currdir, 'solve_with_store8.out'), 'r') as out, open(join(currdir, 'solve_with_store4.txt'), 'r') as txt:
        self.assertStructuredAlmostEqual(yaml.full_load(txt), yaml.full_load(out), allow_second_superset=True)