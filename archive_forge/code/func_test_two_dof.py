from sympy import solve
from sympy.core.backend import (cos, expand, Matrix, sin, symbols, tan, sqrt, S,
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
from sympy.testing.pytest import raises
from sympy.core.backend import USE_SYMENGINE
def test_two_dof():
    q1, q2, u1, u2 = dynamicsymbols('q1 q2 u1 u2')
    q1d, q2d, u1d, u2d = dynamicsymbols('q1 q2 u1 u2', 1)
    m, c1, c2, k1, k2 = symbols('m c1 c2 k1 k2')
    N = ReferenceFrame('N')
    P1 = Point('P1')
    P2 = Point('P2')
    P1.set_vel(N, u1 * N.x)
    P2.set_vel(N, (u1 + u2) * N.x)
    kd = [q1d / 2 - u1 / 2, 2 * q2d - 2 * u2]
    FL = [(P1, (-k1 * q1 - c1 * u1 + k2 * q2 + c2 * u2) * N.x), (P2, (-k2 * q2 - c2 * u2) * N.x)]
    pa1 = Particle('pa1', P1, m)
    pa2 = Particle('pa2', P2, m)
    BL = [pa1, pa2]
    KM = KanesMethod(N, q_ind=[q1, q2], u_ind=[u1, u2], kd_eqs=kd)
    KM.kanes_equations(BL, FL)
    MM = KM.mass_matrix
    forcing = KM.forcing
    rhs = MM.inv() * forcing
    assert expand(rhs[0]) == expand((-k1 * q1 - c1 * u1 + k2 * q2 + c2 * u2) / m)
    assert expand(rhs[1]) == expand((k1 * q1 + c1 * u1 - 2 * k2 * q2 - 2 * c2 * u2) / m)
    assert KM.explicit_kinematics
    assert KM.mass_matrix_kin == eye(2)
    KM.explicit_kinematics = False
    assert KM.mass_matrix_kin == Matrix([[S(1) / 2, 0], [0, 2]])
    for explicit_kinematics in [False, True]:
        KM.explicit_kinematics = explicit_kinematics
        assert simplify(KM.rhs() - KM.mass_matrix_full.LUsolve(KM.forcing_full)) == zeros(4, 1)
    kd = [q1d - u1 ** 2, sin(q2d) - cos(u2)]
    raises(ValueError, lambda: KanesMethod(N, q_ind=[q1, q2], u_ind=[u1, u2], kd_eqs=kd))