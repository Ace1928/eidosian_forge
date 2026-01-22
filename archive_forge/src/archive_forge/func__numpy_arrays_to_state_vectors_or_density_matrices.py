from typing import Optional, TYPE_CHECKING, Tuple
import numpy as np
from cirq import protocols, value, _import
from cirq.qis.states import (
def _numpy_arrays_to_state_vectors_or_density_matrices(state1: np.ndarray, state2: np.ndarray, qid_shape: Optional[Tuple[int, ...]], validate: bool, atol: float) -> Tuple[np.ndarray, np.ndarray]:
    if state1.ndim > 2 or (state1.ndim == 2 and state1.shape[0] != state1.shape[1]):
        state1 = state1.reshape(-1)
    if state2.ndim > 2 or (state2.ndim == 2 and state2.shape[0] != state2.shape[1]):
        state2 = state2.reshape(-1)
    if state1.ndim == 2 and state2.ndim == 2:
        if state1.shape == state2.shape:
            if qid_shape is None:
                raise ValueError('The qid shape of the given states is ambiguous. Try specifying the qid shape explicitly or using a wrapper function like cirq.density_matrix.')
            if state1.shape == qid_shape:
                state1 = state1.reshape(-1)
                state2 = state2.reshape(-1)
        elif state1.shape[0] < state2.shape[0]:
            state1 = state1.reshape(-1)
        else:
            state2 = state2.reshape(-1)
    elif state1.ndim == 2 and state2.ndim < 2 and (np.prod(state1.shape, dtype=np.int64) == np.prod(state2.shape, dtype=np.int64)):
        state1 = state1.reshape(-1)
    elif state1.ndim < 2 and state2.ndim == 2 and (np.prod(state1.shape, dtype=np.int64) == np.prod(state2.shape, dtype=np.int64)):
        state2 = state2.reshape(-1)
    if validate:
        dim1: int = state1.shape[0] if state1.ndim == 2 else np.prod(state1.shape, dtype=np.int64).item()
        dim2: int = state2.shape[0] if state2.ndim == 2 else np.prod(state2.shape, dtype=np.int64).item()
        if dim1 != dim2:
            raise ValueError(f'Mismatched dimensions in given states: {dim1} and {dim2}.')
        if qid_shape is None:
            qid_shape = (dim1,)
        else:
            expected_dim = np.prod(qid_shape, dtype=np.int64)
            if dim1 != expected_dim:
                raise ValueError(f'Invalid state dimension for given qid shape: Expected dimension {expected_dim} but got dimension {dim1}.')
        for state in (state1, state2):
            if state.ndim == 2:
                validate_density_matrix(state, qid_shape=qid_shape, atol=atol)
            else:
                validate_normalized_state_vector(state, qid_shape=qid_shape, atol=atol)
    return (state1, state2)