from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Derivative, Function
from sympy.core.numbers import (Number, pi, I)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import _sympify, sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan2, cos, sin)
from sympy.physics.units import speed_of_light, meter, second
def _eval_rewrite_as_pde(self, *args, **kwargs):
    mu, epsilon, x, t = symbols('mu, epsilon, x, t')
    E = Function('E')
    return Derivative(E(x, t), x, 2) + mu * epsilon * Derivative(E(x, t), t, 2)