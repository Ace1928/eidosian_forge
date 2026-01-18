from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.vector.printing import (VectorLatexPrinter, vpprint,
def test_vlatex():
    from sympy.physics.vector import vlatex
    x = symbols('x')
    J = symbols('J')
    f = Function('f')
    g = Function('g')
    h = Function('h')
    expected = 'J \\left(\\frac{d}{d x} g{\\left(x \\right)} - \\frac{d}{d x} h{\\left(x \\right)}\\right)'
    expr = J * f(x).diff(x).subs(f(x), g(x) - h(x))
    assert vlatex(expr) == expected