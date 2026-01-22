from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.simplify import simplify
from sympy.matrices import zeros
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import pretty_symbol
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.operator import (HermitianOperator, Operator,
from sympy.physics.quantum.state import Bra, Ket, State
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.hilbert import ComplexSpace, DirectSumHilbertSpace
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.qapply import qapply
def _eval_wignerd(self):
    j = self.j
    m = self.m
    mp = self.mp
    alpha = self.alpha
    beta = self.beta
    gamma = self.gamma
    if alpha == 0 and beta == 0 and (gamma == 0):
        return KroneckerDelta(m, mp)
    if not j.is_number:
        raise ValueError('j parameter must be numerical to evaluate, got %s' % j)
    r = 0
    if beta == pi / 2:
        for k in range(2 * j + 1):
            if k > j + mp or k > j - m or k < mp - m:
                continue
            r += S.NegativeOne ** k * binomial(j + mp, k) * binomial(j - mp, k + m - mp)
        r *= S.NegativeOne ** (m - mp) / 2 ** j * sqrt(factorial(j + m) * factorial(j - m) / (factorial(j + mp) * factorial(j - mp)))
    else:
        size, mvals = m_values(j)
        for mpp in mvals:
            r += Rotation.d(j, m, mpp, pi / 2).doit() * (cos(-mpp * beta) + I * sin(-mpp * beta)) * Rotation.d(j, mpp, -mp, pi / 2).doit()
        r = r * I ** (2 * j - m - mp) * (-1) ** (2 * m)
        r = simplify(r)
    r *= exp(-I * m * alpha) * exp(-I * mp * gamma)
    return r