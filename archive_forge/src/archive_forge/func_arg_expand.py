from .plot import BaseSeries, Plot
from .experimental_lambdify import experimental_lambdify, vectorized_lambdify
from .intervalmath import interval
from sympy.core.relational import (Equality, GreaterThan, LessThan,
from sympy.core.containers import Tuple
from sympy.core.relational import Eq
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.logic.boolalg import BooleanFunction
from sympy.polys.polyutils import _sort_gens
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import flatten
import warnings
def arg_expand(bool_expr):
    """
        Recursively expands the arguments of an Boolean Function
        """
    for arg in bool_expr.args:
        if isinstance(arg, BooleanFunction):
            arg_expand(arg)
        elif isinstance(arg, Relational):
            arg_list.append(arg)