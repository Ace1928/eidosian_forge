from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, qis, value
from cirq.value import big_endian_int_to_digits, random_state
def apply_h(self, axis: int, exponent: float=1, global_shift: float=0):
    if exponent % 2 != 0:
        if exponent % 1 != 0:
            raise ValueError('H exponent must be integer')
        t = self.s ^ self.G[axis, :] & self.v
        u = self.s ^ self.F[axis, :] & ~self.v ^ self.M[axis, :] & self.v
        alpha = sum(self.G[axis, :] & ~self.v & self.s) % 2
        beta = sum(self.M[axis, :] & ~self.v & self.s)
        beta += sum(self.F[axis, :] & self.v & self.M[axis, :])
        beta += sum(self.F[axis, :] & self.v & self.s)
        beta %= 2
        delta = (self.gamma[axis] + 2 * (alpha + beta)) % 4
        self.update_sum(t, u, delta=delta, alpha=alpha)
    self.omega *= _phase(exponent, global_shift)