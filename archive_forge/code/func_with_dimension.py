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
def with_dimension(self, dimension: int) -> Qid:
    """Returns a copy with a different dimension or number of levels."""
    return self.qubit.with_dimension(dimension)