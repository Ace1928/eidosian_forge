from sympy.core.backend import zeros, Matrix, diff, eye
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import (ReferenceFrame, dynamicsymbols,
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.mechanics.functions import (
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable
def to_linearizer(self):
    """Returns an instance of the Linearizer class, initiated from the
        data in the KanesMethod class. This may be more desirable than using
        the linearize class method, as the Linearizer object will allow more
        efficient recalculation (i.e. about varying operating points)."""
    if self._fr is None or self._frstar is None:
        raise ValueError('Need to compute Fr, Fr* first.')
    f_c = self._f_h
    if self._f_nh and self._k_nh:
        f_v = self._f_nh + self._k_nh * Matrix(self.u)
    else:
        f_v = Matrix()
    if self._f_dnh and self._k_dnh:
        f_a = self._f_dnh + self._k_dnh * Matrix(self._udot)
    else:
        f_a = Matrix()
    u_zero = {i: 0 for i in self.u}
    ud_zero = {i: 0 for i in self._udot}
    qd_zero = {i: 0 for i in self._qdot}
    qd_u_zero = {i: 0 for i in Matrix([self._qdot, self.u])}
    f_0 = msubs(self._f_k, u_zero) + self._k_kqdot * Matrix(self._qdot)
    f_1 = msubs(self._f_k, qd_zero) + self._k_ku * Matrix(self.u)
    f_2 = msubs(self._frstar, qd_u_zero)
    f_3 = msubs(self._frstar, ud_zero) + self._fr
    f_4 = zeros(len(f_2), 1)
    q = self.q
    u = self.u
    if self._qdep:
        q_i = q[:-len(self._qdep)]
    else:
        q_i = q
    q_d = self._qdep
    if self._udep:
        u_i = u[:-len(self._udep)]
    else:
        u_i = u
    u_d = self._udep
    uaux = self._uaux
    uauxdot = uaux.diff(dynamicsymbols._t)
    uaux_zero = {i: 0 for i in Matrix([uaux, uauxdot])}
    sym_list = set(Matrix([q, self._qdot, u, self._udot, uaux, uauxdot]))
    if any((find_dynamicsymbols(i, sym_list) for i in [self._k_kqdot, self._k_ku, self._f_k, self._k_dnh, self._f_dnh, self._k_d])):
        raise ValueError('Cannot have dynamicsymbols outside dynamic                              forcing vector.')
    r = list(find_dynamicsymbols(msubs(self._f_d, uaux_zero), sym_list))
    r.sort(key=default_sort_key)
    for i in r:
        if diff(i, dynamicsymbols._t) in r:
            raise ValueError('Cannot have derivatives of specified                                  quantities when linearizing forcing terms.')
    return Linearizer(f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a, q, u, q_i, q_d, u_i, u_d, r)