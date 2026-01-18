import pyomo.common.unittest as unittest
from pyomo.common.dependencies import (
from pyomo.environ import (
from pyomo.core.expr import MonomialTermExpression
from pyomo.core.expr.ndarray import NumericNDArray
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.expr.compare import compare_expressions
from pyomo.core.expr.relational_expr import InequalityExpression
from pyomo.repn import generate_standard_repn
def test_numpy_scalar_times_scalar_var(self):
    m = ConcreteModel()
    m.x = Var()
    e = np.float64(5) * m.x
    self.assertIs(type(e), MonomialTermExpression)
    self.assertTrue(compare_expressions(e, 5.0 * m.x))
    e = m.x * np.float64(5)
    self.assertIs(type(e), MonomialTermExpression)
    self.assertTrue(compare_expressions(e, 5.0 * m.x))