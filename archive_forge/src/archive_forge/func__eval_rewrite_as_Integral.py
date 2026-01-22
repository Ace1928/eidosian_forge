from sympy.core import S
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.symbol import Dummy
from sympy.functions.special.gamma_functions import gamma, digamma
from sympy.functions.combinatorial.numbers import catalan
from sympy.functions.elementary.complexes import conjugate
def _eval_rewrite_as_Integral(self, a, b, x1, x2, **kwargs):
    from sympy.integrals.integrals import Integral
    t = Dummy('t')
    integrand = t ** (a - 1) * (1 - t) ** (b - 1)
    expr = Integral(integrand, (t, x1, x2))
    return expr / Integral(integrand, (t, 0, 1))