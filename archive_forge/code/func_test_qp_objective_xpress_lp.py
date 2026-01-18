import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, SolverFactory
@unittest.skipUnless(xpress_lp.available(exception_flag=False), 'needs Xpress LP interface')
def test_qp_objective_xpress_lp(self):
    m = self._qp_model()
    results = xpress_lp.solve(m)
    self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])