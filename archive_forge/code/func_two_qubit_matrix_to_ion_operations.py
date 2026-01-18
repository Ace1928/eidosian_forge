from typing import Iterable, List, Optional, cast, Tuple, TYPE_CHECKING
import numpy as np
from cirq import ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions, two_qubit_to_cz
def two_qubit_matrix_to_ion_operations(q0: 'cirq.Qid', q1: 'cirq.Qid', mat: np.ndarray, atol: float=1e-08, clean_operations: bool=True) -> List[ops.Operation]:
    """Decomposes a two-qubit operation into MS/single-qubit rotation gates.

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        mat: Defines the operation to apply to the pair of qubits.
        atol: A limit on the amount of error introduced by the construction.
        clean_operations: Enables optimizing resulting operation list by
            merging operations and ejecting phased Paulis and Z operations.

    Returns:
        A list of operations implementing the matrix.
    """
    kak = linalg.kak_decomposition(mat, atol=atol)
    operations = _kak_decomposition_to_operations(q0, q1, kak, atol)
    return two_qubit_to_cz.cleanup_operations(operations) if clean_operations else operations