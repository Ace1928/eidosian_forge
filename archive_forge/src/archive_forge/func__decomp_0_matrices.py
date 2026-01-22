from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phxz
def _decomp_0_matrices(kak: 'cirq.KakDecomposition', atol: float=1e-08) -> Tuple[Sequence[Tuple[np.ndarray, np.ndarray]], complex]:
    """Returns the single-qubit matrices for the 0-SQRT_ISWAP decomposition.

    Assumes canonical x, y, z and (x, y, z) = (0, 0, 0) within tolerance.
    """
    return ([(kak.single_qubit_operations_after[0] @ kak.single_qubit_operations_before[0], kak.single_qubit_operations_after[1] @ kak.single_qubit_operations_before[1])], kak.global_phase)