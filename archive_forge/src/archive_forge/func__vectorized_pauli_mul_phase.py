import abc
import numbers
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, linalg, value
from cirq._compat import proper_repr
from cirq.ops import raw_types, identity, pauli_gates, global_phase_op, pauli_string
from cirq.type_workarounds import NotImplementedType
def _vectorized_pauli_mul_phase(lhs: Union[int, np.ndarray], rhs: Union[int, np.ndarray]) -> complex:
    """Computes the leading coefficient of a pauli string multiplication.

    The two inputs must have the same length. They must follow the convention
    that I=0, X=1, Z=2, Y=3 and have no out-of-range values.

    Args:
        lhs: Left hand side `pauli_mask` from `DensePauliString`.
        rhs: Right hand side `pauli_mask` from `DensePauliString`.

    Returns:
        1, 1j, -1, or -1j.
    """
    t = np.array(rhs, dtype=np.int8)
    t *= lhs != 0
    t -= lhs * (rhs != 0)
    t += 1
    t %= 3
    t -= 1
    s = int(np.sum(t, dtype=np.uint8).item() & 3)
    return 1j ** s