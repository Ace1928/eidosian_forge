import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, SolverFactory
@unittest.skipUnless(xpress_nl.available(exception_flag=False), 'needs Xpress NL interface')
def test_qp_objective_xpress_nl(self):
    m = self._qp_model()
    results = xpress_nl.solve(m)
    self.assertIn(str(int(m.obj())), results['Solver'][0]['Message'])