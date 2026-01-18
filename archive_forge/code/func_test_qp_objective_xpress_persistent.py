import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, SolverFactory
@unittest.skipUnless(xpress_persistent.available(exception_flag=False), 'needs Xpress Persistent interface')
def test_qp_objective_xpress_persistent(self):
    m = self._qp_model()
    xpress_persistent.set_instance(m)
    results = xpress_persistent.solve(m)
    self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])