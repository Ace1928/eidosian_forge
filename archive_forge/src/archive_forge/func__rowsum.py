import abc
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING
import numpy as np
from cirq import protocols
from cirq._compat import proper_repr
from cirq.qis import quantum_state_representation
from cirq.value import big_endian_int_to_digits, linear_dict, random_state
def _rowsum(self, q1, q2):
    """Implements the "rowsum" routine defined by
        Aaronson and Gottesman.
        Multiplies the stabilizer in row q1 by the stabilizer in row q2."""

    def g(x1, z1, x2, z2):
        if not x1 and (not z1):
            return 0
        elif x1 and z1:
            return int(z2) - int(x2)
        elif x1 and (not z1):
            return int(z2) * (2 * int(x2) - 1)
        else:
            return int(x2) * (1 - 2 * int(z2))
    r = 2 * int(self._rs[q1]) + 2 * int(self._rs[q2])
    for j in range(self.n):
        r += g(self._xs[q2, j], self._zs[q2, j], self._xs[q1, j], self._zs[q1, j])
    r %= 4
    self._rs[q1] = bool(r)
    self._xs[q1, :] ^= self._xs[q2, :]
    self._zs[q1, :] ^= self._zs[q2, :]