import copy
from io import StringIO
from pyomo.core.expr import expr_common
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.core.base.expression import _GeneralExpressionData
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.common.tee import capture_output
def test_pprint_oldStyle(self):
    expr_common.TO_STRING_VERBOSE = True
    model = ConcreteModel()
    model.x = Var()
    model.e = Expression(initialize=model.x + 2)
    model.E = Expression([1, 2], initialize=model.x ** 2 + 1)
    expr = model.e * model.x ** 2 + model.E[1]
    output = 'sum(prod(e{sum(mon(1, x), 2)}, pow(x, 2)), E[1]{sum(pow(x, 2), 1)})\ne : Size=1, Index=None\n    Key  : Expression\n    None : sum(mon(1, x), 2)\nE : Size=2, Index={1, 2}\n    Key : Expression\n      1 : sum(pow(x, 2), 1)\n      2 : sum(pow(x, 2), 1)\n'
    out = StringIO()
    out.write(str(expr) + '\n')
    model.e.pprint(ostream=out)
    model.E.pprint(ostream=out)
    self.assertEqual(output, out.getvalue())
    model.e.set_value(1.0)
    model.E[1].set_value(2.0)
    output = 'sum(prod(e{1.0}, pow(x, 2)), E[1]{2.0})\ne : Size=1, Index=None\n    Key  : Expression\n    None :        1.0\nE : Size=2, Index={1, 2}\n    Key : Expression\n      1 : 2.0\n      2 : sum(pow(x, 2), 1)\n'
    out = StringIO()
    out.write(str(expr) + '\n')
    model.e.pprint(ostream=out)
    model.E.pprint(ostream=out)
    self.assertEqual(output, out.getvalue())
    model.e.set_value(None)
    model.E[1].set_value(None)
    output = 'sum(prod(e{Undefined}, pow(x, 2)), E[1]{Undefined})\ne : Size=1, Index=None\n    Key  : Expression\n    None :  Undefined\nE : Size=2, Index={1, 2}\n    Key : Expression\n      1 : Undefined\n      2 : sum(pow(x, 2), 1)\n'
    out = StringIO()
    out.write(str(expr) + '\n')
    model.e.pprint(ostream=out)
    model.E.pprint(ostream=out)
    self.assertEqual(output, out.getvalue())