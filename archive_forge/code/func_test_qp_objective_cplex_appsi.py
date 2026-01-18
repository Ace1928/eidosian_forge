import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, SolverFactory
@unittest.skipUnless(cplex_appsi.available(exception_flag=False), 'needs Cplex APPSI interface')
def test_qp_objective_cplex_appsi(self):
    m = self._qp_model()
    cplex_appsi.set_instance(m)
    results = cplex_appsi.solve(m)
    self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])