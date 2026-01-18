import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, SolverFactory
@unittest.skipUnless(cplex_direct.available(exception_flag=False), 'needs Cplex Direct interface')
def test_qp_objective_cplex_direct(self):
    m = self._qp_model()
    results = cplex_direct.solve(m)
    self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])