import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
def with_coefficient(self, new_coefficient: 'cirq.TParamValComplex') -> 'PauliString':
    """Returns a new `PauliString` with `self.coefficient` replaced with `new_coefficient`."""
    return PauliString(qubit_pauli_map=dict(self._qubit_pauli_map), coefficient=new_coefficient)