from typing import Iterator, List, Optional
import itertools
import math
import numpy as np
import cirq
from cirq_google import ops
def two_qubit_matrix_to_sycamore_operations(q0: cirq.Qid, q1: cirq.Qid, mat: np.ndarray, *, atol: float=1e-08, clean_operations: bool=True) -> cirq.OP_TREE:
    """Decomposes a two-qubit unitary matrix into `cirq_google.SYC` + single qubit rotations.

    The analytical decomposition first Synthesizes the given operation using `cirq.CZPowGate` +
    single qubit rotations and then decomposes each `cirq.CZPowGate` into `cirq_google.SYC` +
    single qubit rotations using `cirq_google.known_2q_op_to_sycamore_operations`.

    Note that the resulting decomposition may not be optimal, and users should first try to
    decompose a given operation using `cirq_google.known_2q_op_to_sycamore_operations`.

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        mat: Defines the operation to apply to the pair of qubits.
        atol: A limit on the amount of absolute error introduced by the
            construction.
        clean_operations: Merges runs of single qubit gates to a single `cirq.PhasedXZGate` in
            the resulting operations list.

    Returns:
        A `cirq.OP_TREE` that implements the given unitary operation using only `cirq_google.SYC` +
        single qubit rotations.
    """
    decomposed_ops: List[cirq.OP_TREE] = []
    for op in cirq.two_qubit_matrix_to_cz_operations(q0, q1, mat, allow_partial_czs=True, atol=atol, clean_operations=clean_operations):
        if cirq.num_qubits(op) == 2:
            decomposed_cphase = known_2q_op_to_sycamore_operations(op)
            assert decomposed_cphase is not None
            decomposed_ops.append(decomposed_cphase)
        else:
            decomposed_ops.append(op)
    return [*cirq.merge_single_qubit_gates_to_phxz(cirq.Circuit(decomposed_ops)).all_operations()] if clean_operations else decomposed_ops