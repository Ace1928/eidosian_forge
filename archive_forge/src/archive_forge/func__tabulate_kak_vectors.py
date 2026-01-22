from functools import reduce
from typing import List, NamedTuple, Sequence, Tuple
from dataclasses import dataclass
import numpy as np
import cirq
from cirq import value
from cirq._compat import proper_repr, proper_eq
from cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils import (
def _tabulate_kak_vectors(*, already_tabulated: np.ndarray, base_gate: np.ndarray, max_dist: float, kak_mesh: np.ndarray, local_unitary_pairs: Sequence[_SingleQubitGatePair]) -> _TabulationStepResult:
    """Tabulate KAK vectors from products of local unitaries with a base gate.

    Args:
        already_tabulated: Record of which KAK vectors have already been
            tabulated. kak_mesh[i] has been calculated if i is in tabulation.
        base_gate: The base 2 qubit gate used in the gate product.
        max_dist: The largest allowed Pauli error between a generated 2Q
           unitary and a KAK vector mesh point that it is tabulated to.
        kak_mesh: Sequence of KAK vectors filling the Weyl chamber whose
            nearest neighbor distance is about 2*max_error.
        local_unitary_pairs: Sequence of 2-tuples of single qubit unitary
            tensors, each of shape (N,2,2).

    Returns:
        The newly tabulated KAK vectors and the local unitaries used to generate
        them. This function also updates already_tabulated to include the
        indices of these vectors (within kak_mesh).
    """
    shapes = {pair[0].shape for pair in local_unitary_pairs}
    shapes.update({pair[0].shape for pair in local_unitary_pairs})
    assert len(shapes) == 1
    assert len(shapes.pop()) == 3
    local_cycles = np.array([vector_kron(*pairs) for pairs in local_unitary_pairs])
    prods = np.einsum('ab,...bc,cd', base_gate, local_cycles[0], base_gate)
    for local_cycle in local_cycles[1:]:
        np.einsum('ab,...bc,...cd', base_gate, local_cycle, prods, out=prods)
    kak_vectors = cirq.kak_vector(prods, check_preconditions=False)
    kept_kaks = []
    kept_cycles = []
    for ind, vec in enumerate(kak_vectors):
        dists = np.sqrt(np.sum((kak_mesh - vec) ** 2, axis=-1))
        close = (dists < max_dist).nonzero()[0]
        assert close.shape[0] in (0, 1), f'close.shape: {close.shape}'
        cycles_for_gate = tuple(((k_0[ind], k_1[ind]) for k_0, k_1 in local_unitary_pairs))
        if not np.all(already_tabulated[close]):
            already_tabulated[close] = True
            kept_kaks.append(vec)
            kept_cycles.append(cycles_for_gate)
    return _TabulationStepResult(kept_kaks, kept_cycles)