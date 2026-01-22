from sympy.assumptions import ask, Q
from sympy.core.basic import Basic
from sympy.core.sympify import _sympify
class AssumptionsWrapper(Basic):
    """
    Wrapper over ``Basic`` instances to call predicate query by
    ``.is_[...]`` property

    Parameters
    ==========

    expr : Basic

    assumptions : Boolean, optional

    Examples
    ========

    >>> from sympy import Q, Symbol
    >>> from sympy.assumptions.wrapper import AssumptionsWrapper
    >>> x = Symbol('x', even=True)
    >>> AssumptionsWrapper(x).is_integer
    True
    >>> y = Symbol('y')
    >>> AssumptionsWrapper(y, Q.even(y)).is_integer
    True

    With ``AssumptionsWrapper``, both evaluation and refinement can be supported
    by single implementation.

    >>> from sympy import Function
    >>> class MyAbs(Function):
    ...     @classmethod
    ...     def eval(cls, x, assumptions=True):
    ...         _x = AssumptionsWrapper(x, assumptions)
    ...         if _x.is_nonnegative:
    ...             return x
    ...         if _x.is_negative:
    ...             return -x
    ...     def _eval_refine(self, assumptions):
    ...         return MyAbs.eval(self.args[0], assumptions)
    >>> MyAbs(x)
    MyAbs(x)
    >>> MyAbs(x).refine(Q.positive(x))
    x
    >>> MyAbs(Symbol('y', negative=True))
    -y

    """

    def __new__(cls, expr, assumptions=None):
        if assumptions is None:
            return expr
        obj = super().__new__(cls, expr, _sympify(assumptions))
        obj.expr = expr
        obj.assumptions = assumptions
        return obj
    _eval_is_algebraic = make_eval_method('algebraic')
    _eval_is_antihermitian = make_eval_method('antihermitian')
    _eval_is_commutative = make_eval_method('commutative')
    _eval_is_complex = make_eval_method('complex')
    _eval_is_composite = make_eval_method('composite')
    _eval_is_even = make_eval_method('even')
    _eval_is_extended_negative = make_eval_method('extended_negative')
    _eval_is_extended_nonnegative = make_eval_method('extended_nonnegative')
    _eval_is_extended_nonpositive = make_eval_method('extended_nonpositive')
    _eval_is_extended_nonzero = make_eval_method('extended_nonzero')
    _eval_is_extended_positive = make_eval_method('extended_positive')
    _eval_is_extended_real = make_eval_method('extended_real')
    _eval_is_finite = make_eval_method('finite')
    _eval_is_hermitian = make_eval_method('hermitian')
    _eval_is_imaginary = make_eval_method('imaginary')
    _eval_is_infinite = make_eval_method('infinite')
    _eval_is_integer = make_eval_method('integer')
    _eval_is_irrational = make_eval_method('irrational')
    _eval_is_negative = make_eval_method('negative')
    _eval_is_noninteger = make_eval_method('noninteger')
    _eval_is_nonnegative = make_eval_method('nonnegative')
    _eval_is_nonpositive = make_eval_method('nonpositive')
    _eval_is_nonzero = make_eval_method('nonzero')
    _eval_is_odd = make_eval_method('odd')
    _eval_is_polar = make_eval_method('polar')
    _eval_is_positive = make_eval_method('positive')
    _eval_is_prime = make_eval_method('prime')
    _eval_is_rational = make_eval_method('rational')
    _eval_is_real = make_eval_method('real')
    _eval_is_transcendental = make_eval_method('transcendental')
    _eval_is_zero = make_eval_method('zero')