from sympy.core.backend import Matrix, eye, zeros
from sympy.core.symbol import Dummy
from sympy.utilities.iterables import flatten
from sympy.physics.vector import dynamicsymbols
from sympy.physics.mechanics.functions import msubs
from collections import namedtuple
from collections.abc import Iterable
def _form_block_matrices(self):
    """Form the block matrices for composing M, A, and B."""
    l, m, n, o, s, k = self._dims
    if n != 0:
        self._M_qq = self.f_0.jacobian(self._qd)
        self._A_qq = -(self.f_0 + self.f_1).jacobian(self.q)
    else:
        self._M_qq = Matrix()
        self._A_qq = Matrix()
    if n != 0 and m != 0:
        self._M_uqc = self.f_a.jacobian(self._qd_dup)
        self._A_uqc = -self.f_a.jacobian(self.q)
    else:
        self._M_uqc = Matrix()
        self._A_uqc = Matrix()
    if n != 0 and o - m + k != 0:
        self._M_uqd = self.f_3.jacobian(self._qd_dup)
        self._A_uqd = -(self.f_2 + self.f_3 + self.f_4).jacobian(self.q)
    else:
        self._M_uqd = Matrix()
        self._A_uqd = Matrix()
    if o != 0 and m != 0:
        self._M_uuc = self.f_a.jacobian(self._ud)
        self._A_uuc = -self.f_a.jacobian(self.u)
    else:
        self._M_uuc = Matrix()
        self._A_uuc = Matrix()
    if o != 0 and o - m + k != 0:
        self._M_uud = self.f_2.jacobian(self._ud)
        self._A_uud = -(self.f_2 + self.f_3).jacobian(self.u)
    else:
        self._M_uud = Matrix()
        self._A_uud = Matrix()
    if o != 0 and n != 0:
        self._A_qu = -self.f_1.jacobian(self.u)
    else:
        self._A_qu = Matrix()
    if k != 0 and o - m + k != 0:
        self._M_uld = self.f_4.jacobian(self.lams)
    else:
        self._M_uld = Matrix()
    if s != 0 and o - m + k != 0:
        self._B_u = -self.f_3.jacobian(self.r)
    else:
        self._B_u = Matrix()