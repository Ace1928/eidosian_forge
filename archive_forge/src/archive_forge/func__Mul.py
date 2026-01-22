from sympy.testing.pytest import raises, XFAIL
from sympy.external import import_module
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.numbers import (E, oo)
from sympy.core.power import Pow
from sympy.core.relational import (GreaterThan, LessThan, StrictGreaterThan, StrictLessThan, Unequality)
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import (Abs, conjugate)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import (ceiling, floor)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, csc, sec, sin, tan)
from sympy.integrals.integrals import Integral
from sympy.series.limits import Limit
from sympy.core.relational import Eq, Ne, Lt, Le, Gt, Ge
from sympy.physics.quantum.state import Bra, Ket
from sympy.abc import x, y, z, a, b, c, t, k, n
def _Mul(a, b):
    return Mul(a, b, evaluate=False)