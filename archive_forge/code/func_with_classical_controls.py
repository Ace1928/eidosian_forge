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
def with_classical_controls(self, *conditions: Union[str, 'cirq.MeasurementKey', 'cirq.Condition', sympy.Expr]) -> 'cirq.Operation':
    if not conditions:
        return self
    return self.sub_operation.with_classical_controls(*conditions)