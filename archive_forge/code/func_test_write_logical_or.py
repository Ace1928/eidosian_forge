import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_write_logical_or(self):
    m = self.get_model()
    m.c = LogicalConstraint(expr=m.b.lor(m.i.is_present))
    visitor = self.get_visitor()
    expr = visitor.walk_expression((m.c.expr, m.c, 0))
    self.assertIn(id(m.b), visitor.var_map)
    self.assertIn(id(m.i), visitor.var_map)
    b = visitor.var_map[id(m.b)]
    i = visitor.var_map[id(m.i)]
    self.assertTrue(expr[1].equals(cp.logical_or(b, cp.presence_of(i))))