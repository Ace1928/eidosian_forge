from collections.abc import Callable
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import arity, Function
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.printing.latex import latex
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from .experimental_lambdify import (vectorized_lambdify, lambdify)
from sympy.plotting.textplot import textplot
def _uniform_sampling(self):
    param = self.get_parameter_points()
    fx = vectorized_lambdify([self.var], self.expr_x)
    fy = vectorized_lambdify([self.var], self.expr_y)
    list_x = fx(param)
    list_y = fy(param)
    return (list_x, list_y)