from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j, wigner_9j
from sympy.printing.precedence import PRECEDENCE
def _check_varsh_872_9(term_list):
    a, alpha, alphap, b, beta, betap, c, gamma, lt = map(Wild, ('a', 'alpha', 'alphap', 'b', 'beta', 'betap', 'c', 'gamma', 'lt'))
    expr = lt * CG(a, alpha, b, beta, c, gamma) ** 2
    simp = S.One
    sign = lt / abs(lt)
    x = abs(a - b)
    y = abs(alpha + beta)
    build_expr = a + b + 1 - Piecewise((x, x > y), (0, Eq(x, y)), (y, y > x))
    index_expr = a + b - c
    term_list, other1 = _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, b, beta, c, gamma, lt), (a, alpha, b, beta), build_expr, index_expr)
    x = abs(a - b)
    y = a + b
    build_expr = (y + 1 - x) * (x + y + 1)
    index_expr = (c - x) * (x + c) + c + gamma
    term_list, other2 = _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, b, beta, c, gamma, lt), (a, alpha, b, beta), build_expr, index_expr)
    expr = CG(a, alpha, b, beta, c, gamma) * CG(a, alphap, b, betap, c, gamma)
    simp = KroneckerDelta(alpha, alphap) * KroneckerDelta(beta, betap)
    sign = S.One
    x = abs(a - b)
    y = abs(alpha + beta)
    build_expr = a + b + 1 - Piecewise((x, x > y), (0, Eq(x, y)), (y, y > x))
    index_expr = a + b - c
    term_list, other3 = _check_cg_simp(expr, simp, sign, S.One, term_list, (a, alpha, alphap, b, beta, betap, c, gamma), (a, alpha, alphap, b, beta, betap), build_expr, index_expr)
    x = abs(a - b)
    y = a + b
    build_expr = (y + 1 - x) * (x + y + 1)
    index_expr = (c - x) * (x + c) + c + gamma
    term_list, other4 = _check_cg_simp(expr, simp, sign, S.One, term_list, (a, alpha, alphap, b, beta, betap, c, gamma), (a, alpha, alphap, b, beta, betap), build_expr, index_expr)
    return (term_list, other1 + other2 + other4)