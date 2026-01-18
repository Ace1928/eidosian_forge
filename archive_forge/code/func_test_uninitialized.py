from pyomo.common import unittest
from pyomo.contrib import appsi
import pyomo.environ as pe
from pyomo.core.base.var import ScalarVar
def test_uninitialized(self):
    res = appsi.base.Results()
    self.assertIsNone(res.best_feasible_objective)
    self.assertIsNone(res.best_objective_bound)
    self.assertEqual(res.termination_condition, appsi.base.TerminationCondition.unknown)
    with self.assertRaisesRegex(RuntimeError, '.*does not currently have a valid solution.*'):
        res.solution_loader.load_vars()
    with self.assertRaisesRegex(RuntimeError, '.*does not currently have valid duals.*'):
        res.solution_loader.get_duals()
    with self.assertRaisesRegex(RuntimeError, '.*does not currently have valid reduced costs.*'):
        res.solution_loader.get_reduced_costs()
    with self.assertRaisesRegex(RuntimeError, '.*does not currently have valid slacks.*'):
        res.solution_loader.get_slacks()