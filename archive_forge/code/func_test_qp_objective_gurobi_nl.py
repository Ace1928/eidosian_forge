import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, SolverFactory
@unittest.skipUnless(gurobi_nl.available(exception_flag=False), 'needs Gurobi NL interface')
def test_qp_objective_gurobi_nl(self):
    m = self._qp_model()
    results = gurobi_nl.solve(m)
    self.assertIn(str(int(m.obj())), results['Solver'][0]['Message'])