from sympy.core.backend import cos, Matrix, sin, zeros, tan, pi, symbols
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.solvers.solvers import solve
from sympy.physics.mechanics import (cross, dot, dynamicsymbols,
def test_sub_qdot():
    q1, q2, q3 = dynamicsymbols('q1:4')
    q1d, q2d, q3d = dynamicsymbols('q1:4', level=1)
    u1, u2, u3 = dynamicsymbols('u1:4')
    u_prime, R, M, g, e, f, theta = symbols("u' R, M, g, e, f, theta")
    a, b, mA, mB, IA, J, K, t = symbols('a b mA mB IA J K t')
    IA22, IA23, IA33 = symbols('IA22 IA23 IA33')
    Q1, Q2, Q3 = symbols('Q1 Q2 Q3')
    F = ReferenceFrame('F')
    P = F.orientnew('P', 'axis', [-theta, F.y])
    A = P.orientnew('A', 'axis', [q1, P.x])
    A.set_ang_vel(F, u1 * A.x + u3 * A.z)
    B = A.orientnew('B', 'axis', [q2, A.z])
    C = A.orientnew('C', 'axis', [q3, A.z])
    pD = Point('D')
    pD.set_vel(A, 0)
    pD.set_vel(F, u2 * A.y)
    pS_star = pD.locatenew('S*', e * A.y)
    pQ = pD.locatenew('Q', f * A.y - R * A.x)
    pA_star = pD.locatenew('A*', a * A.y)
    pB_star = pD.locatenew('B*', b * A.z)
    pC_star = pD.locatenew('C*', -b * A.z)
    for p in [pS_star, pQ, pA_star, pB_star, pC_star]:
        p.v2pt_theory(pD, F, A)
    pB_hat = pB_star.locatenew('B^', -R * A.x)
    pC_hat = pC_star.locatenew('C^', -R * A.x)
    pB_hat.v2pt_theory(pB_star, F, B)
    pC_hat.v2pt_theory(pC_star, F, C)
    kde = [dot(p.vel(F), A.y) for p in [pB_hat, pC_hat]]
    kde += [u1 - q1d]
    kde_map = solve(kde, [q1d, q2d, q3d])
    for k, v in list(kde_map.items()):
        kde_map[k.diff(t)] = v.diff(t)
    inertia_A = inertia(A, IA, IA22, IA33, 0, IA23, 0)
    inertia_B = inertia(B, K, K, J)
    inertia_C = inertia(C, K, K, J)
    rbA = RigidBody('rbA', pA_star, A, mA, (inertia_A, pA_star))
    rbB = RigidBody('rbB', pB_star, B, mB, (inertia_B, pB_star))
    rbC = RigidBody('rbC', pC_star, C, mB, (inertia_C, pC_star))
    km = KanesMethod(F, [q1, q2, q3], [u1, u2], kd_eqs=kde, u_auxiliary=[u3])
    forces = [(pS_star, -M * g * F.x), (pQ, Q1 * A.x + Q2 * A.y + Q3 * A.z)]
    bodies = [rbA, rbB, rbC]
    fr_expected = Matrix([f * Q3 + M * g * e * sin(theta) * cos(q1), Q2 + M * g * sin(theta) * sin(q1), e * M * g * cos(theta) - Q1 * f - Q2 * R])
    fr_star_expected = Matrix([-(IA + 2 * J * b ** 2 / R ** 2 + 2 * K + mA * a ** 2 + 2 * mB * b ** 2) * u1.diff(t) - mA * a * u1 * u2, -(mA + 2 * mB + 2 * J / R ** 2) * u2.diff(t) + mA * a * u1 ** 2, 0])
    fr, fr_star = km.kanes_equations(bodies, forces)
    assert fr.expand() == fr_expected.expand()
    assert (fr_star_expected - trigsimp(fr_star)).expand() == zeros(3, 1)