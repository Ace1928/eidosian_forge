from typing import Any, TypeVar, Optional, Sequence, Union
import numpy as np
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols import unitary_protocol
def _strat_from_trace_distance_bound_method(val: Any) -> Optional[float]:
    """Attempts to use a specialized method."""
    getter = getattr(val, '_trace_distance_bound_', None)
    result = NotImplemented if getter is None else getter()
    if result is None:
        return None
    if result is not NotImplemented:
        return min(1.0, result)
    return NotImplemented