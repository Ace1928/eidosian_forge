from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def _potential_qid_shapes(state: _NON_INT_STATE_LIKE) -> '_QidShapeSet':
    """Return a set of qid shapes compatible with a given state."""
    if isinstance(state, QuantumState):
        return _QidShapeSet(explicit_qid_shapes={state.qid_shape})
    if isinstance(state, value.ProductState):
        return _QidShapeSet(explicit_qid_shapes={(2,) * len(state)})
    if isinstance(state, Sequence):
        state = np.array(state)
    if state.ndim == 1:
        dim, = state.shape
        min_qudit_dimensions = None
        if state.dtype.kind[0] in '?bBiu':
            min_qudit_dimensions = tuple(state.astype(int, copy=False) + 1)
        return _QidShapeSet(unfactorized_total_dimension=dim, min_qudit_dimensions=min_qudit_dimensions)
    if state.ndim == 2:
        dim, _ = state.shape
        return _QidShapeSet(explicit_qid_shapes={state.shape}, unfactorized_total_dimension=dim)
    return _QidShapeSet(explicit_qid_shapes={state.shape})