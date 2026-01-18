from typing import Iterable, List, Sequence, Tuple, Optional, cast, TYPE_CHECKING
import numpy as np
from cirq.linalg import predicates
from cirq.linalg.decompositions import num_cnots_required, extract_right_diag
from cirq import ops, linalg, protocols, circuits
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phased_x_and_z
from cirq.transformers.eject_z import eject_z
from cirq.transformers.eject_phased_paulis import eject_phased_paulis
def two_qubit_matrix_to_diagonal_and_cz_operations(q0: 'cirq.Qid', q1: 'cirq.Qid', mat: np.ndarray, allow_partial_czs: bool=False, atol: float=1e-08, clean_operations: bool=True) -> Tuple[np.ndarray, List['cirq.Operation']]:
    """Decomposes a 2-qubit unitary to a diagonal and the remaining operations.

    For a 2-qubit unitary V, return ops, a list of operations and
    D diagonal unitary, so that:
        V = cirq.Circuit(ops) @ D

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        mat: the input unitary
        allow_partial_czs: Enables the use of Partial-CZ gates.
        atol: A limit on the amount of absolute error introduced by the
            construction.
        clean_operations: Enables optimizing resulting operation list by
            merging operations and ejecting phased Paulis and Z operations.
    Returns:
        tuple(ops,D): operations `ops`, and the diagonal `D`
    """
    if predicates.is_diagonal(mat, atol=atol):
        return (mat, [])
    if num_cnots_required(mat) == 3:
        right_diag = extract_right_diag(mat)
        two_cnot_unitary = mat @ right_diag
        return (right_diag.conj().T, two_qubit_matrix_to_cz_operations(q0, q1, two_cnot_unitary, allow_partial_czs=allow_partial_czs, atol=atol, clean_operations=clean_operations))
    return (np.eye(4), two_qubit_matrix_to_cz_operations(q0, q1, mat, allow_partial_czs=allow_partial_czs, atol=atol, clean_operations=clean_operations))