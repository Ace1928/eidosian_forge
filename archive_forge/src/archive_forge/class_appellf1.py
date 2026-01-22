from functools import reduce
from sympy.core import S, ilcm, Mod
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Function, Derivative, ArgumentIndexError
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.numbers import I, pi, oo, zoo
from sympy.core.relational import Ne
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.functions import (sqrt, exp, log, sin, cos, asin, atan,
from sympy.functions import factorial, RisingFactorial
from sympy.functions.elementary.complexes import Abs, re, unpolarify
from sympy.functions.elementary.exponential import exp_polar
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import (And, Or)
class appellf1(Function):
    """
    This is the Appell hypergeometric function of two variables as:

    .. math ::
        F_1(a,b_1,b_2,c,x,y) = \\sum_{m=0}^{\\infty} \\sum_{n=0}^{\\infty}
        \\frac{(a)_{m+n} (b_1)_m (b_2)_n}{(c)_{m+n}}
        \\frac{x^m y^n}{m! n!}.

    Examples
    ========

    >>> from sympy import appellf1, symbols
    >>> x, y, a, b1, b2, c = symbols('x y a b1 b2 c')
    >>> appellf1(2., 1., 6., 4., 5., 6.)
    0.0063339426292673
    >>> appellf1(12., 12., 6., 4., 0.5, 0.12)
    172870711.659936
    >>> appellf1(40, 2, 6, 4, 15, 60)
    appellf1(40, 2, 6, 4, 15, 60)
    >>> appellf1(20., 12., 10., 3., 0.5, 0.12)
    15605338197184.4
    >>> appellf1(40, 2, 6, 4, x, y)
    appellf1(40, 2, 6, 4, x, y)
    >>> appellf1(a, b1, b2, c, x, y)
    appellf1(a, b1, b2, c, x, y)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Appell_series
    .. [2] https://functions.wolfram.com/HypergeometricFunctions/AppellF1/

    """

    @classmethod
    def eval(cls, a, b1, b2, c, x, y):
        if default_sort_key(b1) > default_sort_key(b2):
            b1, b2 = (b2, b1)
            x, y = (y, x)
            return cls(a, b1, b2, c, x, y)
        elif b1 == b2 and default_sort_key(x) > default_sort_key(y):
            x, y = (y, x)
            return cls(a, b1, b2, c, x, y)
        if x == 0 and y == 0:
            return S.One

    def fdiff(self, argindex=5):
        a, b1, b2, c, x, y = self.args
        if argindex == 5:
            return a * b1 / c * appellf1(a + 1, b1 + 1, b2, c + 1, x, y)
        elif argindex == 6:
            return a * b2 / c * appellf1(a + 1, b1, b2 + 1, c + 1, x, y)
        elif argindex in (1, 2, 3, 4):
            return Derivative(self, self.args[argindex - 1])
        else:
            raise ArgumentIndexError(self, argindex)