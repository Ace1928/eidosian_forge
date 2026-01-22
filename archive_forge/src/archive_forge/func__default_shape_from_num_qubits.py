import abc
import functools
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, value
from cirq._import import LazyLoader
from cirq._compat import __cirq_debug__, cached_method
from cirq.type_workarounds import NotImplementedType
from cirq.ops import control_values as cv
def _default_shape_from_num_qubits(self) -> Tuple[int, ...]:
    num_qubits = self._num_qubits_()
    if num_qubits is NotImplemented:
        return NotImplemented
    return (2,) * num_qubits