import os
from pyomo.environ import (
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
@unittest.skipIf(not cbc_available, 'The CBC solver is not available')
def test_rc_signs(self):
    m = ConcreteModel()
    m.x = Var(bounds=(-1, 1))
    m.obj = Objective(expr=m.x)
    m.rc = Suffix(direction=Suffix.IMPORT)
    opt = SolverFactory('cbc')
    res = opt.solve(m)
    self.assertAlmostEqual(res.problem.lower_bound, -1)
    self.assertAlmostEqual(res.problem.upper_bound, -1)
    self.assertAlmostEqual(m.rc[m.x], 1)
    m.obj.sense = maximize
    res = opt.solve(m)
    self.assertAlmostEqual(res.problem.lower_bound, 1)
    self.assertAlmostEqual(res.problem.upper_bound, 1)
    self.assertAlmostEqual(m.rc[m.x], 1)