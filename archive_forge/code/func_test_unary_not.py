import operator
from itertools import permutations, product
import pyomo.common.unittest as unittest
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import (
def test_unary_not(self):
    m = ConcreteModel()
    m.Y = BooleanVar()
    op_static = lnot(m.Y)
    op_operator = ~m.Y
    for truth_combination in _generate_possible_truth_inputs(1):
        m.Y.set_value(truth_combination[0])
        correct_value = not truth_combination[0]
        self.assertEqual(value(op_static), correct_value)
        self.assertEqual(value(op_operator), correct_value)