from sympy.core.function import Derivative
from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.simplify import simplify
from sympy.core.symbol import symbols
from sympy.core import S
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.vector.vector import Dot
from sympy.vector.operators import curl, divergence, gradient, Gradient, Divergence, Cross
from sympy.vector.deloperator import Del
from sympy.vector.functions import (is_conservative, is_solenoidal,
from sympy.testing.pytest import raises
def test_product_rules():
    """
    Tests the six product rules defined with respect to the Del
    operator

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Del

    """
    f = 2 * x * y * z
    g = x * y + y * z + z * x
    u = x ** 2 * i + 4 * j - y ** 2 * z * k
    v = 4 * i + x * y * z * k
    lhs = delop(f * g, doit=True)
    rhs = (f * delop(g) + g * delop(f)).doit()
    assert simplify(lhs) == simplify(rhs)
    lhs = delop(u & v).doit()
    rhs = ((u ^ (delop ^ v)) + (v ^ (delop ^ u)) + (u & delop)(v) + (v & delop)(u)).doit()
    assert simplify(lhs) == simplify(rhs)
    lhs = (delop & f * v).doit()
    rhs = (f * (delop & v) + (v & delop(f))).doit()
    assert simplify(lhs) == simplify(rhs)
    lhs = (delop & (u ^ v)).doit()
    rhs = ((v & (delop ^ u)) - (u & (delop ^ v))).doit()
    assert simplify(lhs) == simplify(rhs)
    lhs = (delop ^ f * v).doit()
    rhs = ((delop(f) ^ v) + f * (delop ^ v)).doit()
    assert simplify(lhs) == simplify(rhs)
    lhs = (delop ^ (u ^ v)).doit()
    rhs = (u * (delop & v) - v * (delop & u) + (v & delop)(u) - (u & delop)(v)).doit()
    assert simplify(lhs) == simplify(rhs)