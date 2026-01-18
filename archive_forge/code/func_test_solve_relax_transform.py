import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import check_available_solvers
@unittest.skipIf(len(solvers) == 0, 'LP/MIP solver not available')
def test_solve_relax_transform(self):
    s = SolverFactory(solvers[0])
    m = _generateModel()
    self.assertIs(m.x.domain, Binary)
    self.assertEqual(m.x.lb, 0)
    self.assertEqual(m.x.ub, 1)
    s.solve(m)
    self.assertEqual(len(m.dual), 0)
    TransformationFactory('core.relax_discrete').apply_to(m)
    self.assertIs(m.x.domain, Reals)
    self.assertEqual(m.x.lb, 0)
    self.assertEqual(m.x.ub, 1)
    s.solve(m)
    self.assertEqual(len(m.dual), 2)
    self.assertAlmostEqual(m.dual[m.c1], -0.5, 4)
    self.assertAlmostEqual(m.dual[m.c2], -0.5, 4)