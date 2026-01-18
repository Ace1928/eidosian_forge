import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, SolverFactory
@unittest.skipUnless(cplex_nl.available(exception_flag=False), 'needs Cplex NL interface')
def test_qp_objective_cplex_nl(self):
    m = self._qp_model()
    results = cplex_nl.solve(m)
    self.assertIn(str(int(m.obj())), results['Solver'][0]['Message'])