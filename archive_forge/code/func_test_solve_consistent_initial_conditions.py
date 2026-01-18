import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.initialization import (
@unittest.skipIf(not ipopt_available, 'ipopt is not available')
def test_solve_consistent_initial_conditions(self):
    m = make_model()
    solver = SolverFactory('ipopt')
    solve_consistent_initial_conditions(m, m.time, solver, allow_skip=True)
    inconsistent = get_inconsistent_initial_conditions(m, m.time)
    self.assertFalse(inconsistent)
    self.assertTrue(m.fs.con1[m.time[1]].active)
    self.assertTrue(m.fs.con1[m.time[3]].active)
    self.assertTrue(m.fs.b1.con[m.time[1], m.space[1]].active)
    self.assertTrue(m.fs.b1.con[m.time[3], m.space[1]].active)
    with self.assertRaises(KeyError):
        solve_consistent_initial_conditions(m, m.time, solver, allow_skip=False)