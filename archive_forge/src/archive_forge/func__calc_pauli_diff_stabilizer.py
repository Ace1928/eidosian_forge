from __future__ import annotations
from collections.abc import Callable
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states import StabilizerState
from qiskit.synthesis.linear.linear_matrix_utils import (
from qiskit.synthesis.linear_phase import synth_cz_depth_line_mr
from qiskit.synthesis.clifford.clifford_decompose_layers import (
def _calc_pauli_diff_stabilizer(cliff, cliff_target):
    """Given two Cliffords whose stabilizers differ by a Pauli, we find this Pauli."""
    from qiskit.quantum_info.operators.symplectic import Pauli
    num_qubits = cliff.num_qubits
    if cliff.num_qubits != cliff_target.num_qubits:
        raise QiskitError('num_qubits is not the same for the original clifford and the target.')
    stab_gen = StabilizerState(cliff).clifford.to_dict()['stabilizer']
    ts = StabilizerState(cliff_target)
    phase_destab = [False] * num_qubits
    phase_stab = [ts.expectation_value(Pauli(stab_gen[i])) == -1 for i in range(num_qubits)]
    phase = []
    phase.extend(phase_destab)
    phase.extend(phase_stab)
    phase = np.array(phase, dtype=int)
    A = cliff.symplectic_matrix.astype(int)
    Ainv = calc_inverse_matrix(A)
    C = np.matmul(Ainv, phase) % 2
    pauli_circ = QuantumCircuit(num_qubits, name='Pauli')
    for k in range(num_qubits):
        destab = C[k]
        stab = C[k + num_qubits]
        if stab and destab:
            pauli_circ.y(k)
        elif stab:
            pauli_circ.x(k)
        elif destab:
            pauli_circ.z(k)
    return pauli_circ