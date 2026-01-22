from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phxz
def _decomp_to_operations(q0: 'cirq.Qid', q1: 'cirq.Qid', two_qubit_gate: 'cirq.Gate', single_qubit_operations: Sequence[Tuple[np.ndarray, np.ndarray]], u0_before: np.ndarray=np.eye(2), u0_after: np.ndarray=np.eye(2), atol: float=1e-08) -> Sequence['cirq.Operation']:
    """Converts a sequence of single-qubit unitary matrices on two qubits into a
    list of operations with interleaved two-qubit gates."""
    two_qubit_op = two_qubit_gate(q0, q1)
    operations = []
    prev_commute = 1

    def append(matrix0, matrix1, final_layer=False):
        """Appends the decomposed single-qubit operations for matrix0 and
        matrix1.

        The cleanup logic, specific to sqrt-iSWAP, commutes the final Z**a gate
        and any whole X or Y gate on q1 through the following sqrt-iSWAP.

        Commutation rules:
        - Z(q0)**a, Z(q1)**a together commute with sqrt-iSWAP for all a
        - X(q0), X(q0) together commute with sqrt-iSWAP
        - Y(q0), Y(q0) together commute with sqrt-iSWAP
        """
        nonlocal prev_commute
        rots1 = list(single_qubit_decompositions.single_qubit_matrix_to_pauli_rotations(np.dot(matrix1, prev_commute), atol=atol))
        new_commute = np.eye(2, dtype=matrix0.dtype)
        if not final_layer:
            if len(rots1) > 0 and rots1[-1][0] == ops.Z:
                _, prev_z = rots1.pop()
                z_unitary = protocols.unitary(ops.Z ** prev_z)
                new_commute = new_commute @ z_unitary
                matrix0 = z_unitary.T.conj() @ matrix0
            if len(rots1) > 0 and linalg.tolerance.near_zero_mod(rots1[-1][1], 1, atol=atol):
                pauli, half_turns = rots1.pop()
                p_unitary = protocols.unitary(pauli ** half_turns)
                new_commute = new_commute @ p_unitary
                matrix0 = p_unitary.T.conj() @ matrix0
        rots0 = list(single_qubit_decompositions.single_qubit_matrix_to_pauli_rotations(np.dot(matrix0, prev_commute), atol=atol))
        operations.extend(((pauli ** half_turns).on(q0) for pauli, half_turns in rots0))
        operations.extend(((pauli ** half_turns).on(q1) for pauli, half_turns in rots1))
        prev_commute = new_commute
    single_ops = list(single_qubit_operations)
    if len(single_ops) <= 1:
        for matrix0, matrix1 in single_ops:
            append(matrix0, matrix1, final_layer=True)
        return operations
    for matrix0, matrix1 in single_ops[:1]:
        append(u0_before @ matrix0, matrix1)
        operations.append(two_qubit_op)
    for matrix0, matrix1 in single_ops[1:-1]:
        append(u0_before @ matrix0 @ u0_after, matrix1)
        operations.append(two_qubit_op)
    for matrix0, matrix1 in single_ops[-1:]:
        append(matrix0 @ u0_after, matrix1, final_layer=True)
    return operations