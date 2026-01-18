import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, SolverFactory
@unittest.skipUnless(xpress_direct.available(exception_flag=False), 'needs Xpress Direct interface')
def test_qp_objective_xpress_direct(self):
    m = self._qp_model()
    results = xpress_direct.solve(m)
    self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])