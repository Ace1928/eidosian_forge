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
def to_z_basis_ops(self) -> Iterator[raw_types.Operation]:
    """Returns single qubit operations to convert the qubits to the computational basis."""
    for qubit, pauli in self.items():
        yield clifford_gate.SingleQubitCliffordGate.from_single_map({pauli: (pauli_gates.Z, False)})(qubit)