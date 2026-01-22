from sympy.core.backend import zeros, Matrix, diff, eye
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import (ReferenceFrame, dynamicsymbols,
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.mechanics.functions import (
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable
def _form_frstar(self, bl):
    """Form the generalized inertia force."""
    if not iterable(bl):
        raise TypeError('Bodies must be supplied in an iterable.')
    t = dynamicsymbols._t
    N = self._inertial
    udot_zero = {i: 0 for i in self._udot}
    uaux_zero = {i: 0 for i in self._uaux}
    uauxdot = [diff(i, t) for i in self._uaux]
    uauxdot_zero = {i: 0 for i in uauxdot}
    q_ddot_u_map = {k.diff(t): v.diff(t) for k, v in self._qdot_u_map.items()}
    q_ddot_u_map.update(self._qdot_u_map)

    def get_partial_velocity(body):
        if isinstance(body, RigidBody):
            vlist = [body.masscenter.vel(N), body.frame.ang_vel_in(N)]
        elif isinstance(body, Particle):
            vlist = [body.point.vel(N)]
        else:
            raise TypeError('The body list may only contain either RigidBody or Particle as list elements.')
        v = [msubs(vel, self._qdot_u_map) for vel in vlist]
        return partial_velocity(v, self.u, N)
    partials = [get_partial_velocity(body) for body in bl]
    o = len(self.u)
    MM = zeros(o, o)
    nonMM = zeros(o, 1)
    zero_uaux = lambda expr: msubs(expr, uaux_zero)
    zero_udot_uaux = lambda expr: msubs(msubs(expr, udot_zero), uaux_zero)
    for i, body in enumerate(bl):
        if isinstance(body, RigidBody):
            M = zero_uaux(body.mass)
            I = zero_uaux(body.central_inertia)
            vel = zero_uaux(body.masscenter.vel(N))
            omega = zero_uaux(body.frame.ang_vel_in(N))
            acc = zero_udot_uaux(body.masscenter.acc(N))
            inertial_force = M.diff(t) * vel + M * acc
            inertial_torque = zero_uaux((I.dt(body.frame) & omega) + msubs(I & body.frame.ang_acc_in(N), udot_zero) + (omega ^ I & omega))
            for j in range(o):
                tmp_vel = zero_uaux(partials[i][0][j])
                tmp_ang = zero_uaux(I & partials[i][1][j])
                for k in range(o):
                    MM[j, k] += M * (tmp_vel & partials[i][0][k])
                    MM[j, k] += tmp_ang & partials[i][1][k]
                nonMM[j] += inertial_force & partials[i][0][j]
                nonMM[j] += inertial_torque & partials[i][1][j]
        else:
            M = zero_uaux(body.mass)
            vel = zero_uaux(body.point.vel(N))
            acc = zero_udot_uaux(body.point.acc(N))
            inertial_force = M.diff(t) * vel + M * acc
            for j in range(o):
                temp = zero_uaux(partials[i][0][j])
                for k in range(o):
                    MM[j, k] += M * (temp & partials[i][0][k])
                nonMM[j] += inertial_force & partials[i][0][j]
    MM = zero_uaux(msubs(MM, q_ddot_u_map))
    nonMM = msubs(msubs(nonMM, q_ddot_u_map), udot_zero, uauxdot_zero, uaux_zero)
    fr_star = -(MM * msubs(Matrix(self._udot), uauxdot_zero) + nonMM)
    if self._udep:
        p = o - len(self._udep)
        fr_star_ind = fr_star[:p, 0]
        fr_star_dep = fr_star[p:o, 0]
        fr_star = fr_star_ind + self._Ars.T * fr_star_dep
        MMi = MM[:p, :]
        MMd = MM[p:o, :]
        MM = MMi + self._Ars.T * MMd
    self._bodylist = bl
    self._frstar = fr_star
    self._k_d = MM
    self._f_d = -msubs(self._fr + self._frstar, udot_zero)
    return fr_star