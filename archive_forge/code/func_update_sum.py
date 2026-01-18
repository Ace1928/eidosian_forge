from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, qis, value
from cirq.value import big_endian_int_to_digits, random_state
def update_sum(self, t, u, delta=0, alpha=0):
    """Implements the transformation (Proposition 4 in Bravyi et al)

        ``i^alpha U_H (|t> + i^delta |u>) = omega W_C W_H |s'>``
        """
    if np.all(t == u):
        self.s = t
        self.omega *= 1 / np.sqrt(2) * (-1) ** alpha * (1 + 1j ** delta)
        return
    set0 = np.where(~self.v & (t ^ u))[0]
    set1 = np.where(self.v & (t ^ u))[0]
    if len(set0) > 0:
        q = set0[0]
        for i in set0:
            if i != q:
                self._CNOT_right(q, i)
        for i in set1:
            self._CZ_right(q, i)
    elif len(set1) > 0:
        q = set1[0]
        for i in set1:
            if i != q:
                self._CNOT_right(i, q)
    e = np.zeros(self.n, dtype=bool)
    e[q] = True
    if t[q]:
        y = u ^ e
        z = u
    else:
        y = t
        z = t ^ e
    omega, a, b, c = self._H_decompose(self.v[q], y[q], z[q], delta)
    self.s = y
    self.s[q] = c
    self.omega *= (-1) ** alpha * omega
    if a:
        self._S_right(q)
    self.v[q] ^= b ^ self.v[q]