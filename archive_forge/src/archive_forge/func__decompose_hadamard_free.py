from __future__ import annotations
from collections.abc import Callable
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford  # pylint: disable=cyclic-import
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
from qiskit.synthesis.linear import (
from qiskit.synthesis.linear_phase import synth_cz_depth_line_mr, synth_cx_cz_depth_line_my
from qiskit.synthesis.linear.linear_matrix_utils import (
def _decompose_hadamard_free(cliff, validate, cz_synth_func, cx_synth_func, cx_cz_synth_func, cz_func_reverse_qubits):
    """Assumes that the Clifford cliff is Hadamard free.
    Decompose it into the layers S2 - CZ2 - CX, where
    S2_circ is a circuit that can contain only S gates,
    CZ2_circ is a circuit that can contain only CZ gates, and
    CX_circ is a circuit that can contain CX gates.

    Args:
        cliff (Clifford): a Hadamard-free clifford operator.
        validate (Boolean): if True, validates the synthesis process.
        cz_synth_func (Callable): a function to decompose the CZ sub-circuit.
        cx_synth_func (Callable): a function to decompose the CX sub-circuit.
        cx_cz_synth_func (Callable): optional, a function to decompose both sub-circuits CZ and CX.
        cz_func_reverse_qubits (Boolean): True only if cz_synth_func is synth_cz_depth_line_mr.

    Returns:
        S2_circ: a circuit that can contain only S gates.
        CZ2_circ: a circuit that can contain only CZ gates.
        CX_circ: a circuit that can contain only CX gates.

    Raises:
        QiskitError: if cliff is not Hadamard free.
    """
    num_qubits = cliff.num_qubits
    destabx = cliff.destab_x
    destabz = cliff.destab_z
    stabx = cliff.stab_x
    if not (stabx == np.zeros((num_qubits, num_qubits))).all():
        raise QiskitError('The given Clifford is not Hadamard-free.')
    destabz_update = np.matmul(calc_inverse_matrix(destabx), destabz) % 2
    if validate:
        if (destabz_update != destabz_update.T).any():
            raise QiskitError('The multiplication of the inverse of destabx anddestabz is not a symmetric matrix.')
    S2_circ = QuantumCircuit(num_qubits, name='S2')
    for i in range(0, num_qubits):
        if destabz_update[i][i]:
            S2_circ.s(i)
    if cx_cz_synth_func is not None:
        for i in range(num_qubits):
            destabz_update[i][i] = 0
        mat_z = destabz_update
        mat_x = calc_inverse_matrix(destabx.transpose())
        CXCZ_circ = cx_cz_synth_func(mat_x, mat_z)
        return (S2_circ, QuantumCircuit(num_qubits), CXCZ_circ)
    CZ2_circ = cz_synth_func(destabz_update)
    mat = destabx.transpose()
    if cz_func_reverse_qubits:
        mat = np.flip(mat, axis=0)
    CX_circ = cx_synth_func(mat)
    return (S2_circ, CZ2_circ, CX_circ)