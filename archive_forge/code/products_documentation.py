from typing import Tuple as tTuple
from .expr_with_intlimits import ExprWithIntLimits
from .summations import Sum, summation, _dummy_with_inherited_properties_concrete
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import Derivative
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.combinatorial.factorials import RisingFactorial
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.polys import quo, roots

        Reverse the order of a limit in a Product.

        Explanation
        ===========

        ``reverse_order(expr, *indices)`` reverses some limits in the expression
        ``expr`` which can be either a ``Sum`` or a ``Product``. The selectors in
        the argument ``indices`` specify some indices whose limits get reversed.
        These selectors are either variable names or numerical indices counted
        starting from the inner-most limit tuple.

        Examples
        ========

        >>> from sympy import gamma, Product, simplify, Sum
        >>> from sympy.abc import x, y, a, b, c, d
        >>> P = Product(x, (x, a, b))
        >>> Pr = P.reverse_order(x)
        >>> Pr
        Product(1/x, (x, b + 1, a - 1))
        >>> Pr = Pr.doit()
        >>> Pr
        1/RisingFactorial(b + 1, a - b - 1)
        >>> simplify(Pr.rewrite(gamma))
        Piecewise((gamma(b + 1)/gamma(a), b > -1), ((-1)**(-a + b + 1)*gamma(1 - a)/gamma(-b), True))
        >>> P = P.doit()
        >>> P
        RisingFactorial(a, -a + b + 1)
        >>> simplify(P.rewrite(gamma))
        Piecewise((gamma(b + 1)/gamma(a), a > 0), ((-1)**(-a + b + 1)*gamma(1 - a)/gamma(-b), True))

        While one should prefer variable names when specifying which limits
        to reverse, the index counting notation comes in handy in case there
        are several symbols with the same name.

        >>> S = Sum(x*y, (x, a, b), (y, c, d))
        >>> S
        Sum(x*y, (x, a, b), (y, c, d))
        >>> S0 = S.reverse_order(0)
        >>> S0
        Sum(-x*y, (x, b + 1, a - 1), (y, c, d))
        >>> S1 = S0.reverse_order(1)
        >>> S1
        Sum(x*y, (x, b + 1, a - 1), (y, d + 1, c - 1))

        Of course we can mix both notations:

        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(x, 1)
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))
        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(y, x)
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))

        See Also
        ========

        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.index,
        reorder_limit,
        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.reorder

        References
        ==========

        .. [1] Michael Karr, "Summation in Finite Terms", Journal of the ACM,
               Volume 28 Issue 2, April 1981, Pages 305-350
               https://dl.acm.org/doi/10.1145/322248.322255

        