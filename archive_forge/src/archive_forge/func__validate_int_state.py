from typing import Optional, TYPE_CHECKING, Tuple
import numpy as np
from cirq import protocols, value, _import
from cirq.qis.states import (
def _validate_int_state(state: int, qid_shape: Optional[Tuple[int, ...]]) -> None:
    if state < 0:
        raise ValueError(f'Invalid state: A state specified as an integer must be non-negative, but {state} was given.')
    if qid_shape is not None:
        dim = np.prod(qid_shape, dtype=np.int64)
        if state >= dim:
            raise ValueError(f'Invalid state for given qid shape: The maximum computational basis state for qid shape {qid_shape} is {dim - 1}, but {state} was given.')