import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, SolverFactory
@unittest.skipUnless(gurobi_lp.available(exception_flag=False), 'needs Gurobi LP interface')
def test_qp_objective_gurobi_lp(self):
    m = self._qp_model()
    results = gurobi_lp.solve(m)
    self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])