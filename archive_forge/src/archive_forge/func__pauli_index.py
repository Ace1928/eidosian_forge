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
def _pauli_index(val: 'cirq.PAULI_GATE_LIKE') -> int:
    m = pauli_string.PAULI_GATE_LIKE_TO_INDEX_MAP
    if val not in m:
        raise TypeError(f'Expected a cirq.PAULI_GATE_LIKE (any of cirq.I cirq.X, cirq.Y, cirq.Z, "I", "X", "Y", "Z", "i", "x", "y", "z", 0, 1, 2, 3) but got {repr(val)}.')
    return m[val]