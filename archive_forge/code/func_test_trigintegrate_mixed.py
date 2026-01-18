from sympy.core import Ne, Rational, Symbol
from sympy.functions import sin, cos, tan, csc, sec, cot, log, Piecewise
from sympy.integrals.trigonometry import trigintegrate
def test_trigintegrate_mixed():
    assert trigintegrate(sin(x) * sec(x), x) == -log(cos(x))
    assert trigintegrate(sin(x) * csc(x), x) == x
    assert trigintegrate(sin(x) * cot(x), x) == sin(x)
    assert trigintegrate(cos(x) * sec(x), x) == x
    assert trigintegrate(cos(x) * csc(x), x) == log(sin(x))
    assert trigintegrate(cos(x) * tan(x), x) == -cos(x)
    assert trigintegrate(cos(x) * cot(x), x) == log(cos(x) - 1) / 2 - log(cos(x) + 1) / 2 + cos(x)
    assert trigintegrate(cot(x) * cos(x) ** 2, x) == log(sin(x)) - sin(x) ** 2 / 2