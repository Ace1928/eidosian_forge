from typing import List, TYPE_CHECKING
import numpy as np
from cirq import ops
from cirq.transformers.analytical_decompositions import two_qubit_to_cz
def two_qubit_matrix_to_cz_isometry(q0: 'cirq.Qid', q1: 'cirq.Qid', mat: np.ndarray, allow_partial_czs: bool=False, atol: float=1e-08, clean_operations: bool=True) -> List['cirq.Operation']:
    """Decomposes a 2q operation into at-most 2 CZs + 1q rotations; assuming `q0` is initially |0>.

    The method implements isometry from one to two qubits; assuming qubit `q0` is always in the |0>
    state. See Appendix B.1 of https://arxiv.org/abs/1501.06911 for more details.

    Args:
        q0: The first qubit being operated on. This is assumed to always be in the |0> state.
        q1: The other qubit being operated on.
        mat: Defines the unitary operation to apply to the pair of qubits.
        allow_partial_czs: Enables the use of Partial-CZ gates.
        atol: A limit on the amount of absolute error introduced by the construction.
        clean_operations: Enables optimizing resulting operation list by merging single qubit
        operations and ejecting phased Paulis and Z operations.

    Returns:
        A list of operations implementing the action of the given unitary matrix, assuming
        the input qubit `q0` is in the |0> state.
    """
    d, cz_ops = two_qubit_to_cz.two_qubit_matrix_to_diagonal_and_cz_operations(q0, q1, mat, allow_partial_czs, atol, clean_operations)
    decomposed_ops = [ops.PhasedXZGate.from_matrix(np.diag([d[0][0], d[1][1]])).on(q1)] + cz_ops
    return two_qubit_to_cz.cleanup_operations(decomposed_ops) if clean_operations else decomposed_ops