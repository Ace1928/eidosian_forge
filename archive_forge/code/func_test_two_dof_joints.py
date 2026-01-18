from sympy.core.function import expand
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.mechanics import (PinJoint, JointsMethod, Body, KanesMethod,
from sympy.physics.vector import dynamicsymbols, ReferenceFrame
from sympy.testing.pytest import raises
from sympy.core.backend import zeros
from sympy.utilities.lambdify import lambdify
from sympy.solvers.solvers import solve
def test_two_dof_joints():
    q1, q2, u1, u2 = dynamicsymbols('q1 q2 u1 u2')
    m, c1, c2, k1, k2 = symbols('m c1 c2 k1 k2')
    W = Body('W')
    B1 = Body('B1', mass=m)
    B2 = Body('B2', mass=m)
    J1 = PrismaticJoint('J1', W, B1, coordinates=q1, speeds=u1)
    J2 = PrismaticJoint('J2', B1, B2, coordinates=q2, speeds=u2)
    W.apply_force(k1 * q1 * W.x, reaction_body=B1)
    W.apply_force(c1 * u1 * W.x, reaction_body=B1)
    B1.apply_force(k2 * q2 * W.x, reaction_body=B2)
    B1.apply_force(c2 * u2 * W.x, reaction_body=B2)
    method = JointsMethod(W, J1, J2)
    method.form_eoms()
    MM = method.mass_matrix
    forcing = method.forcing
    rhs = MM.LUsolve(forcing)
    assert expand(rhs[0]) == expand((-k1 * q1 - c1 * u1 + k2 * q2 + c2 * u2) / m)
    assert expand(rhs[1]) == expand((k1 * q1 + c1 * u1 - 2 * k2 * q2 - 2 * c2 * u2) / m)