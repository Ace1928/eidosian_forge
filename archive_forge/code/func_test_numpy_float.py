import pyomo.common.unittest as unittest
from pyomo.common.dependencies import (
from pyomo.environ import (
from pyomo.core.expr import MonomialTermExpression
from pyomo.core.expr.ndarray import NumericNDArray
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.expr.compare import compare_expressions
from pyomo.core.expr.relational_expr import InequalityExpression
from pyomo.repn import generate_standard_repn
def test_numpy_float(self):
    m = ConcreteModel()
    m.T = Set(initialize=range(3))
    m.v = Var(initialize=1, bounds=(0, None))
    m.c = Var(m.T, initialize=20)
    h = [np.float32(1.0), 1.0, 1]

    def rule(m, t):
        return m.c[0] == h[t] * m.c[0]
    m.x = Constraint(m.T, rule=rule)

    def rule(m, t):
        return m.c[0] == h[t] * m.c[0] * m.v
    m.y = Constraint(m.T, rule=rule)

    def rule(m, t):
        return m.c[0] == h[t] * m.v
    m.z = Constraint(m.T, rule=rule)
    for t in m.T:
        self.assertTrue(compare_expressions(m.x[0].expr, m.x[t].expr))
        self.assertTrue(compare_expressions(m.y[0].expr, m.y[t].expr))
        self.assertTrue(compare_expressions(m.z[0].expr, m.z[t].expr))