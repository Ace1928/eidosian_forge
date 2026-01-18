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
def wrap_in_linear_combination(self, coefficient: Union[complex, float, int]=1) -> 'cirq.LinearCombinationOfGates':
    """Returns a LinearCombinationOfGates with this gate.

        Args:
            coefficient: number coefficient to use in the resulting
                `cirq.LinearCombinationOfGates` object.

        Returns:
            `cirq.LinearCombinationOfGates` containing self with a
                coefficient of `coefficient`.
        """
    return ops.linear_combinations.LinearCombinationOfGates({self: coefficient})