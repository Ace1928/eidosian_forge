from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
@property
def all_single_qubit_cliffords(cls) -> Sequence['cirq.SingleQubitCliffordGate']:
    """All 24 single-qubit Clifford gates."""
    if not hasattr(cls, '_all_single_qubit_cliffords'):
        pX = (pauli_gates.X, False)
        mX = (pauli_gates.X, True)
        pY = (pauli_gates.Y, False)
        mY = (pauli_gates.Y, True)
        pZ = (pauli_gates.Z, False)
        mZ = (pauli_gates.Z, True)

        def from_xz(x_to, z_to):
            return SingleQubitCliffordGate.from_clifford_tableau(_to_clifford_tableau(x_to=x_to, z_to=z_to))
        cls._all_single_qubit_cliffords = (from_xz(x_to=pX, z_to=pZ), from_xz(x_to=pX, z_to=mZ), from_xz(x_to=mX, z_to=mZ), from_xz(x_to=mX, z_to=pZ), from_xz(x_to=pX, z_to=mY), from_xz(x_to=mZ, z_to=pX), from_xz(x_to=pY, z_to=pZ), from_xz(x_to=pX, z_to=pY), from_xz(x_to=pZ, z_to=mX), from_xz(x_to=mY, z_to=pZ), from_xz(x_to=pZ, z_to=pX), from_xz(x_to=pY, z_to=mZ), from_xz(x_to=mX, z_to=pY), from_xz(x_to=mZ, z_to=mX), from_xz(x_to=mY, z_to=mZ), from_xz(x_to=mX, z_to=mY), from_xz(x_to=pY, z_to=pX), from_xz(x_to=mZ, z_to=mY), from_xz(x_to=pZ, z_to=mY), from_xz(x_to=mY, z_to=mX), from_xz(x_to=mZ, z_to=pY), from_xz(x_to=mY, z_to=pX), from_xz(x_to=pY, z_to=mX), from_xz(x_to=pZ, z_to=pY))
    return cls._all_single_qubit_cliffords