from typing import Any, Union, Iterable
from fractions import Fraction
from decimal import Decimal
import numbers
import numpy as np
import sympy
from typing_extensions import Protocol
from cirq._doc import doc_private
def _isclose(a: Any, b: Any, *, atol: Union[int, float]) -> bool:
    """Convenience wrapper around np.isclose."""
    x1 = np.asarray([a])
    if isinstance(a, (Fraction, Decimal)):
        x1 = x1.astype(np.float64)
    x2 = np.asarray([b])
    if isinstance(b, (Fraction, Decimal)):
        x2 = x2.astype(np.float64)
    try:
        result = bool(np.isclose(x1, x2, atol=atol, rtol=0.0)[0])
    except TypeError:
        return NotImplemented
    return result