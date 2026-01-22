from typing import AbstractSet, Any, Dict, Optional, Tuple, TYPE_CHECKING, Union, List
import datetime
import sympy
import numpy as np
from cirq import protocols
from cirq._compat import proper_repr, cached_method
from cirq._doc import document
def _decompose_into_amount_unit_suffix(self) -> Tuple[int, str, str]:
    picos = self.total_picos()
    if isinstance(picos, sympy.Mul) and len(picos.args) == 2 and isinstance(picos.args[0], (sympy.Integer, sympy.Float)):
        scale = picos.args[0]
        rest = picos.args[1]
    else:
        scale = picos
        rest = 1
    if scale % 1000000000 == 0:
        amount = scale / 1000000000
        unit = 'millis'
        suffix = 'ms'
    elif scale % 1000000 == 0:
        amount = scale / 1000000
        unit = 'micros'
        suffix = 'us'
    elif scale % 1000 == 0:
        amount = scale / 1000
        unit = 'nanos'
        suffix = 'ns'
    else:
        amount = scale
        unit = 'picos'
        suffix = 'ps'
    if isinstance(scale, int):
        amount = int(amount)
    return (amount * rest, unit, suffix)