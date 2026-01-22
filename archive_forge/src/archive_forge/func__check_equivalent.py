import operator
from itertools import permutations, product
import pyomo.common.unittest as unittest
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import (
def _check_equivalent(assert_handle, expr_1, expr_2):
    expr_1_vars = list(identify_variables(expr_1, include_fixed=False))
    expr_2_vars = list(identify_variables(expr_2, include_fixed=False))
    assert_handle.assertEqual(len(expr_1_vars), len(expr_2_vars))
    for truth_combination in _generate_possible_truth_inputs(len(expr_1_vars)):
        for var, truth_value in zip(expr_1_vars, truth_combination):
            var.value = truth_value
        assert_handle.assertEqual(value(expr_1), value(expr_2))