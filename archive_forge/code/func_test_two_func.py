from sympy.external import import_module
from sympy.testing.pytest import raises
import ctypes
import sympy
from sympy.abc import a, b, n
def test_two_func():
    e = 4.0 * sympy.exp(-a) + sympy.exp(b)
    f = g.llvm_callable([a, b], e)
    res = float(e.subs({a: 1.5, b: 2.0}).evalf())
    jit_res = f(1.5, 2.0)
    assert isclose(jit_res, res)