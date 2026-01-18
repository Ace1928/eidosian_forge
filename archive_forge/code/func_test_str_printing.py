from sympy.core.function import Function
from sympy.integrals.integrals import Integral
from sympy.printing.latex import latex
from sympy.printing.pretty import pretty as xpretty
from sympy.vector import CoordSys3D, Del, Vector, express
from sympy.abc import a, b, c
from sympy.testing.pytest import XFAIL
def test_str_printing():
    assert str(v[0]) == '0'
    assert str(v[1]) == 'N.i'
    assert str(v[2]) == '(-1)*N.i'
    assert str(v[3]) == 'N.i + N.j'
    assert str(v[8]) == 'N.j + (C.x**2 - Integral(f(b), b))*N.k'
    assert str(v[9]) == 'C.k + N.i'
    assert str(s) == '3*C.y*N.x**2'
    assert str(d[0]) == '0'
    assert str(d[1]) == '(N.i|N.k)'
    assert str(d[4]) == 'a*(N.i|N.k)'
    assert str(d[5]) == 'a*(N.i|N.k) + (-b)*(N.j|N.k)'
    assert str(d[8]) == '(N.j|N.k) + (C.x**2 - ' + 'Integral(f(b), b))*(N.k|N.k)'