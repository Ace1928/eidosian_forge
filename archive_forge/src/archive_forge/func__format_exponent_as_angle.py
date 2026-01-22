import abc
import fractions
import math
import numbers
from typing import (
import numpy as np
import sympy
from cirq import value, protocols
from cirq.linalg import tolerance
from cirq.ops import raw_types
from cirq.type_workarounds import NotImplementedType
def _format_exponent_as_angle(self, args: 'protocols.CircuitDiagramInfoArgs', order: int=2) -> str:
    """Returns string with exponent expressed as angle in radians.

        Args:
            args: CircuitDiagramInfoArgs describing the desired drawing style.
            order: Exponent corresponding to full rotation by 2π.

        Returns:
            Angle in radians corresponding to the exponent of self and
            formatted according to style described by args.
        """
    exponent = self._diagram_exponent(args, ignore_global_phase=False)
    pi = sympy.pi if protocols.is_parameterized(exponent) else np.pi
    return args.format_radians(radians=2 * pi * exponent / order)