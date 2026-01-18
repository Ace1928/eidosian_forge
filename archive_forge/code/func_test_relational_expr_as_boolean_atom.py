from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
def test_relational_expr_as_boolean_atom(self):
    m = self.make_model()
    m.x = Var()
    e = m.a.land(m.x >= 3)
    visitor = LogicalToDisjunctiveVisitor()
    with self.assertRaisesRegex(MouseTrap, "The RelationalExpression '3 <= x' was used as a Boolean term in a logical proposition. This is not yet supported when transforming to disjunctive form.", normalize_whitespace=True):
        visitor.walk_expression(e)