import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
from .clifford_decompose_bm import _decompose_clifford_1q
def _set_row_z_zero(clifford, circuit, qubit):
    """Set stabilizer.Z[qubit, i] to False for all i > qubit.

    Implemented by applying (reverse) CNOTS assumes qubit < num_qubits
    and _set_row_x_zero has been called first
    """
    x = clifford.stab_x[qubit]
    z = clifford.stab_z[qubit]
    if np.any(z[qubit + 1:]):
        for i in range(qubit + 1, clifford.num_qubits):
            if z[i]:
                _append_cx(clifford, i, qubit)
                circuit.cx(i, qubit)
    if np.any(x[qubit:]):
        _append_h(clifford, qubit)
        circuit.h(qubit)
        for i in range(qubit + 1, clifford.num_qubits):
            if x[i]:
                _append_cx(clifford, qubit, i)
                circuit.cx(qubit, i)
        if z[qubit]:
            _append_s(clifford, qubit)
            circuit.s(qubit)
        _append_h(clifford, qubit)
        circuit.h(qubit)