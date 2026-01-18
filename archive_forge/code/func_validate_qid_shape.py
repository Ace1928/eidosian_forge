from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def validate_qid_shape(state_vector: np.ndarray, qid_shape: Optional[Tuple[int, ...]]) -> Tuple[int, ...]:
    """Validates the size of the given `state_vector` against the given shape.

    Returns:
        The qid shape.

    Raises:
        ValueError: if the size of `state_vector` does not match that given in
            `qid_shape` or if `qid_state` is not given if `state_vector` does
            not have a dimension that is a power of two.
    """
    size = state_vector.size
    if qid_shape is None:
        qid_shape = (2,) * (size.bit_length() - 1)
    if size != np.prod(qid_shape, dtype=np.int64):
        raise ValueError(f'state_vector.size ({size}) is not a power of two or is not a product of the qid shape {qid_shape!r}.')
    return qid_shape