import sys
import types
import inspect
from functools import wraps, update_wrapper
from sympy.utilities.exceptions import sympy_deprecation_warning
def threaded_factory(func, use_add):
    """A factory for ``threaded`` decorators. """
    from sympy.core import sympify
    from sympy.matrices import MatrixBase
    from sympy.utilities.iterables import iterable

    @wraps(func)
    def threaded_func(expr, *args, **kwargs):
        if isinstance(expr, MatrixBase):
            return expr.applyfunc(lambda f: func(f, *args, **kwargs))
        elif iterable(expr):
            try:
                return expr.__class__([func(f, *args, **kwargs) for f in expr])
            except TypeError:
                return expr
        else:
            expr = sympify(expr)
            if use_add and expr.is_Add:
                return expr.__class__(*[func(f, *args, **kwargs) for f in expr.args])
            elif expr.is_Relational:
                return expr.__class__(func(expr.lhs, *args, **kwargs), func(expr.rhs, *args, **kwargs))
            else:
                return func(expr, *args, **kwargs)
    return threaded_func