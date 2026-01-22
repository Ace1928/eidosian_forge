from typing import AbstractSet, Any, Dict, Optional, Tuple, TYPE_CHECKING, Union, List
import datetime
import sympy
import numpy as np
from cirq import protocols
from cirq._compat import proper_repr, cached_method
from cirq._doc import document
def _duration_from_time_vals(time_vals: List[_NUMERIC_INPUT_TYPE]):
    ret = Duration()
    ret._time_vals = time_vals
    return ret