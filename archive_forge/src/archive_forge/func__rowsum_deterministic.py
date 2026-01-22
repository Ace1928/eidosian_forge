from __future__ import annotations
from collections.abc import Collection
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import Clifford, Pauli, PauliList
from qiskit.quantum_info.operators.symplectic.clifford_circuits import _append_x
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.circuit import QuantumCircuit, Instruction
@staticmethod
def _rowsum_deterministic(clifford, aux_pauli, row):
    """Updating an auxilary Pauli aux_pauli in the
        deterministic rowsum calculation.
        The StabilizerState itself is not updated."""
    row_phase = clifford.phase[row]
    accum_phase = aux_pauli.phase
    accum_pauli = aux_pauli
    row_pauli = Pauli((clifford.z[row], clifford.x[row]))
    accum_pauli, accum_phase = StabilizerState._rowsum(accum_pauli, accum_phase, row_pauli, row_phase)
    aux_pauli = accum_pauli
    aux_pauli.phase = accum_phase
    return aux_pauli