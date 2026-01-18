from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def validate_normalized_state_vector(state_vector: np.ndarray, *, qid_shape: Tuple[int, ...], dtype: Optional['DTypeLike']=None, atol: float=1e-07) -> None:
    """Checks that the given state vector is valid.

    Args:
        state_vector: The state vector to validate.
        qid_shape: The expected qid shape of the state.
        dtype: The expected dtype of the state.
        atol: Absolute numerical tolerance.

    Raises:
        ValueError: State has invalid dtype.
        ValueError: State has incorrect size.
        ValueError: State is not normalized.
    """
    if dtype and state_vector.dtype != dtype:
        raise ValueError(f'state_vector has invalid dtype. Expected {dtype} but was {state_vector.dtype}')
    if state_vector.size != np.prod(qid_shape, dtype=np.int64):
        raise ValueError(f'state_vector has incorrect size. Expected {np.prod(qid_shape, dtype=np.int64)} but was {state_vector.size}.')
    norm = np.sum(np.abs(state_vector) ** 2)
    if not np.isclose(norm, 1, atol=atol):
        raise ValueError(f'State_vector is not normalized instead had norm {norm}')