from typing import Iterator, List, Optional
import itertools
import math
import numpy as np
import cirq
from cirq_google import ops
def _find_local_equivalents(target_unitary: np.ndarray, source_unitary: np.ndarray):
    """Determine the local 1q rotations that turn one equivalent 2q unitary into the other.

    Given two 2q unitaries with the same interaction coefficients but different local unitary
    rotations determine the local unitaries that turns one type of gate into another.

    1) Perform the KAK Decomposition on each unitary and confirm interaction terms are equivalent.
    2) Identify the elements of SU(2) to transform source_unitary into target_unitary

    Args:
        target_unitary: The unitary that we need to transform `source_unitary` to.
        source_unitary: The unitary that we need to transform by adding local gates, and make it
            equivalent to the target_unitary.

    Returns:
        Four 2x2 unitaries. The first two are pre-rotations and last two are post rotations.
    """
    kak_u1 = cirq.kak_decomposition(target_unitary)
    kak_u2 = cirq.kak_decomposition(source_unitary)
    u_0 = kak_u1.single_qubit_operations_after[0] @ kak_u2.single_qubit_operations_after[0].conj().T
    u_1 = kak_u1.single_qubit_operations_after[1] @ kak_u2.single_qubit_operations_after[1].conj().T
    v_0 = kak_u2.single_qubit_operations_before[0].conj().T @ kak_u1.single_qubit_operations_before[0]
    v_1 = kak_u2.single_qubit_operations_before[1].conj().T @ kak_u1.single_qubit_operations_before[1]
    return (v_0, v_1, u_0, u_1)