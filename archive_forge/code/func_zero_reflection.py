from __future__ import annotations
from typing import List, Optional, Union
import numpy
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Statevector, Operator, DensityMatrix
from .standard_gates import MCXGate
@property
def zero_reflection(self) -> QuantumCircuit:
    """The subcircuit implementing the reflection about 0."""
    if self._zero_reflection is not None:
        return self._zero_reflection
    num_state_qubits = self.oracle.num_qubits - self.oracle.num_ancillas
    return _zero_reflection(num_state_qubits, self.reflection_qubits, self._mcx_mode)