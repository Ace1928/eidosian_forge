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
def without_classical_controls(self) -> 'cirq.Operation':
    new_sub_operation = self.sub_operation.without_classical_controls()
    return self if new_sub_operation is self.sub_operation else new_sub_operation