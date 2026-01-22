from typing import Any, overload, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def _strat_commutes_from_matrix(v1: Any, v2: Any, *, atol: float) -> Union[bool, NotImplementedType, None]:
    """Attempts to determine commutativity of matrices."""
    if not isinstance(v1, np.ndarray) or not isinstance(v2, np.ndarray):
        return NotImplemented
    if v1.shape != v2.shape:
        return None
    return linalg.matrix_commutes(v1, v2, atol=atol)