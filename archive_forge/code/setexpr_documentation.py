from sympy.core import Expr
from sympy.core.decorators import call_highest_priority, _sympifyit
from .fancysets import ImageSet
from .sets import set_add, set_sub, set_mul, set_div, set_pow, set_function
An expression that can take on values of a set.

    Examples
    ========

    >>> from sympy import Interval, FiniteSet
    >>> from sympy.sets.setexpr import SetExpr

    >>> a = SetExpr(Interval(0, 5))
    >>> b = SetExpr(FiniteSet(1, 10))
    >>> (a + b).set
    Union(Interval(1, 6), Interval(10, 15))
    >>> (2*a + b).set
    Interval(1, 20)
    