from typing import AbstractSet, Any, Dict, Optional, Tuple, TYPE_CHECKING, Union, List
import datetime
import sympy
import numpy as np
from cirq import protocols
from cirq._compat import proper_repr, cached_method
from cirq._doc import document
def total_micros(self) -> _NUMERIC_OUTPUT_TYPE:
    """Returns the number of microseconds that the duration spans."""
    return self.total_picos() / 1000000