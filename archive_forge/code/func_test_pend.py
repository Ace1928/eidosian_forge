from sympy import solve
from sympy.core.backend import (cos, expand, Matrix, sin, symbols, tan, sqrt, S,
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
from sympy.testing.pytest import raises
from sympy.core.backend import USE_SYMENGINE
def test_pend():
    q, u = dynamicsymbols('q u')
    qd, ud = dynamicsymbols('q u', 1)
    m, l, g = symbols('m l g')
    N = ReferenceFrame('N')
    P = Point('P')
    P.set_vel(N, -l * u * sin(q) * N.x + l * u * cos(q) * N.y)
    kd = [qd - u]
    FL = [(P, m * g * N.x)]
    pa = Particle('pa', P, m)
    BL = [pa]
    KM = KanesMethod(N, [q], [u], kd)
    KM.kanes_equations(BL, FL)
    MM = KM.mass_matrix
    forcing = KM.forcing
    rhs = MM.inv() * forcing
    rhs.simplify()
    assert expand(rhs[0]) == expand(-g / l * sin(q))
    assert simplify(KM.rhs() - KM.mass_matrix_full.LUsolve(KM.forcing_full)) == zeros(2, 1)