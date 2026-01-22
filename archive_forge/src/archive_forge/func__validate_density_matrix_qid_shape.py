from typing import List, Optional, TYPE_CHECKING, Tuple, Sequence
import numpy as np
from cirq import linalg, value
from cirq.sim import simulation_utils
def _validate_density_matrix_qid_shape(density_matrix: np.ndarray, qid_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Validates that a tensor's shape is a valid shape for qids and returns the
    qid shape.
    """
    shape = density_matrix.shape
    if len(shape) == 2:
        if np.prod(qid_shape, dtype=np.int64) ** 2 != np.prod(shape, dtype=np.int64):
            raise ValueError(f'Matrix size does not match qid shape {qid_shape!r}. Got matrix with shape {shape!r}. Expected {np.prod(qid_shape, dtype=np.int64)!r}.')
        return qid_shape
    if len(shape) % 2 != 0:
        raise ValueError(f'Tensor was not square. Shape was {shape}')
    left_shape = shape[:len(shape) // 2]
    right_shape = shape[len(shape) // 2:]
    if left_shape != right_shape:
        raise ValueError(f"Tensor's left and right shape are not equal. Shape was {shape}")
    return left_shape