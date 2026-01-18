from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
def test_numeric_expression_in_at_most(self):
    m = self.make_model()
    m.x = Var([1, 2], bounds=(0, 10), domain=Integers)
    m.y = Var(domain=Integers)
    m.e = Expression(expr=m.x[1] * m.x[2])
    e = atmost(m.e + m.y, m.a, m.b, m.c)
    visitor = LogicalToDisjunctiveVisitor()
    m.cons = visitor.constraints
    m.z = visitor.z_vars
    with self.assertRaisesRegex(MouseTrap, "The first argument '\\(x\\[1\\]\\*x\\[2\\]\\) \\+ y' to 'atmost\\(\\(x\\[1\\]\\*x\\[2\\]\\) \\+ y: \\[a, b, c\\]\\)' is potentially variable. This may be a mathematically coherent expression; However it is not yet supported to convert it to a disjunctive program", normalize_whitespace=True):
        visitor.walk_expression(e)