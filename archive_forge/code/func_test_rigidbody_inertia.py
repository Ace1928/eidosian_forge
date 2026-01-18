from sympy.core.symbol import symbols
from sympy.physics.mechanics import Point, ReferenceFrame, Dyadic, RigidBody
from sympy.physics.mechanics import dynamicsymbols, outer, inertia
from sympy.physics.mechanics import inertia_of_point_mass
from sympy.core.backend import expand, zeros, _simplify_matrix
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_rigidbody_inertia():
    N = ReferenceFrame('N')
    m, Ix, Iy, Iz, a, b = symbols('m, I_x, I_y, I_z, a, b')
    Io = inertia(N, Ix, Iy, Iz)
    o = Point('o')
    p = o.locatenew('p', a * N.x + b * N.y)
    R = RigidBody('R', o, N, m, (Io, p))
    I_check = inertia(N, Ix - b ** 2 * m, Iy - a ** 2 * m, Iz - m * (a ** 2 + b ** 2), m * a * b)
    assert R.inertia == (Io, p)
    assert R.central_inertia == I_check
    R.central_inertia = Io
    assert R.inertia == (Io, o)
    assert R.central_inertia == Io
    R.inertia = (Io, p)
    assert R.inertia == (Io, p)
    assert R.central_inertia == I_check