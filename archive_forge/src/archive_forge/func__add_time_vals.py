from typing import AbstractSet, Any, Dict, Optional, Tuple, TYPE_CHECKING, Union, List
import datetime
import sympy
import numpy as np
from cirq import protocols
from cirq._compat import proper_repr, cached_method
from cirq._doc import document
def _add_time_vals(val1: List[_NUMERIC_INPUT_TYPE], val2: List[_NUMERIC_INPUT_TYPE]) -> List[_NUMERIC_INPUT_TYPE]:
    ret: List[_NUMERIC_INPUT_TYPE] = []
    for i in range(4):
        if val1[i] and val2[i]:
            ret.append(val1[i] + val2[i])
        else:
            ret.append(val1[i] or val2[i])
    return ret