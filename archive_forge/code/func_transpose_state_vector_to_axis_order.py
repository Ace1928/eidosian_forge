import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
def transpose_state_vector_to_axis_order(t: np.ndarray, axes: Sequence[int]):
    """Transposes the axes of a state vector to a specified order.

    Args:
        t: The state vector to transpose.
        axes: The desired axis order.
    Returns:
        The transposed state vector.
    """
    assert set(axes) == set(range(int(t.ndim))), 'All axes must be provided.'
    return np.moveaxis(t, axes, range(len(axes)))