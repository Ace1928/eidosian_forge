import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
def _imul_helper(self, other: 'cirq.PAULI_STRING_LIKE', sign: int):
    """Left-multiplies or right-multiplies by a PAULI_STRING_LIKE.

        Args:
            other: What to multiply by.
            sign: +1 to left-multiply, -1 to right-multiply.

        Returns:
            self on success, NotImplemented given an unknown type of value.
        """
    if isinstance(other, (Mapping, PauliString, MutablePauliString)):
        if isinstance(other, (PauliString, MutablePauliString)):
            self.coefficient *= other.coefficient
        phase_log_i = 0
        for qubit, pauli_gate_like in other.items():
            pauli_int = _pauli_like_to_pauli_int(qubit, pauli_gate_like)
            phase_log_i += self._imul_atom_helper(cast(TKey, qubit), pauli_int, sign)
        self.coefficient *= 1j ** (phase_log_i & 3)
    elif isinstance(other, numbers.Number):
        self.coefficient *= complex(cast(SupportsComplex, other))
    elif isinstance(other, raw_types.Operation) and isinstance(other.gate, identity.IdentityGate):
        pass
    elif isinstance(other, Iterable) and (not isinstance(other, str)) and (not isinstance(other, linear_combinations.PauliSum)):
        if sign == +1:
            other = iter(reversed(list(other)))
        for item in other:
            if self._imul_helper(cast(PAULI_STRING_LIKE, item), sign) is NotImplemented:
                return NotImplemented
    else:
        return NotImplemented
    return self