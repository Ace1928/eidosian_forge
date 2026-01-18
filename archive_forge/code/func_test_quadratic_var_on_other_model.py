import os
import random
from ..lp_diff import load_and_compare_lp_baseline
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, Block, ComponentMap
def test_quadratic_var_on_other_model(self):
    baseline_fname, test_fname = self._get_fnames()
    other = ConcreteModel()
    other.a = Var()
    model = ConcreteModel()
    model.x = Var()
    model.obj = Objective(expr=model.x)
    model.c = Constraint(expr=other.a * model.x <= 0)
    with LoggingIntercept() as LOG:
        self.assertRaises(KeyError, model.write, test_fname, format='lp_v1')
    self.assertEqual(LOG.getvalue().replace('\n', ' ').strip(), 'Model contains an expression (c) that contains a variable (a) that is not attached to an active block on the submodel being written')
    model.write(test_fname, format='lp_v2')
    self.assertEqual(*load_and_compare_lp_baseline(baseline_fname, test_fname, 'lp_v2'))