from typing import Optional, TYPE_CHECKING, Tuple
import numpy as np
from cirq import protocols, value, _import
from cirq.qis.states import (
def _fidelity_state_vectors_or_density_matrices(state1: np.ndarray, state2: np.ndarray) -> float:
    if state1.ndim == 1 and state2.ndim == 1:
        return np.abs(np.vdot(state1, state2)) ** 2
    elif state1.ndim == 1 and state2.ndim == 2:
        return np.real(np.conjugate(state1) @ state2 @ state1)
    elif state1.ndim == 2 and state2.ndim == 1:
        return np.real(np.conjugate(state2) @ state1 @ state2)
    elif state1.ndim == 2 and state2.ndim == 2:
        state1_sqrt = _sqrt_positive_semidefinite_matrix(state1)
        eigs = linalg.eigvalsh(state1_sqrt @ state2 @ state1_sqrt)
        trace = np.sum(np.sqrt(np.abs(eigs)))
        return trace ** 2
    raise ValueError(f'The given arrays must be one- or two-dimensional. Got shapes {state1.shape} and {state2.shape}.')