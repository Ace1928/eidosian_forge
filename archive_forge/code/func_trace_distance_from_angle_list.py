from typing import Any, TypeVar, Optional, Sequence, Union
import numpy as np
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols import unitary_protocol
def trace_distance_from_angle_list(angle_list: Union[Sequence[float], np.ndarray]) -> float:
    """Given a list of arguments of the eigenvalues of a unitary matrix,
    calculates the trace distance bound of the unitary effect.

    The maximum provided angle should not exceed the minimum provided angle
    by more than 2π.
    """
    angles = np.sort(angle_list)
    maxim = 2 * np.pi + angles[0] - angles[-1]
    for i in range(1, len(angles)):
        maxim = max(maxim, angles[i] - angles[i - 1])
    if maxim <= np.pi:
        return 1.0
    return max(0.0, np.sin(0.5 * maxim))