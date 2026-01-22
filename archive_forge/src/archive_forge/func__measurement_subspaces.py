from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from collections import defaultdict
import itertools
import random
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols, qis
from cirq.testing import lin_alg_utils
def _measurement_subspaces(measured_qubits: Iterable[ops.Qid], n_qubits: int) -> Sequence[Sequence[int]]:
    """Computes subspaces associated with projective measurement.

    The function computes a partitioning of the computational basis such
    that the subspace spanned by each partition corresponds to a distinct
    measurement outcome. In particular, if all qubits are measured then
    2**n singleton partitions are returned. If no qubits are measured then
    a single partition consisting of all basis states is returned.

    Args:
        measured_qubits: Qubits subject to measurement
        n_qubits: Total number of qubits in circuit
    Returns:
        Sequence of subspaces where each subspace is a sequence of
            computational basis states in order corresponding to qubit_order
    """
    measurement_mask = 0
    for i, _ in enumerate(sorted(measured_qubits)):
        measurement_mask |= 1 << i
    measurement_subspaces: Dict[int, List[int]] = defaultdict(list)
    computational_basis = range(1 << n_qubits)
    for basis_state in computational_basis:
        subspace_key = basis_state & measurement_mask
        measurement_subspaces[subspace_key].append(basis_state)
    subspaces = list(measurement_subspaces.values())
    assert sorted(itertools.chain(*subspaces)) == list(computational_basis)
    return subspaces