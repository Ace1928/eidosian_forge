from __future__ import annotations
import numpy as np
from qiskit.circuit import QuantumCircuit
from .piecewise_linear_pauli_rotations import PiecewiseLinearPauliRotations
Map the function value of the approximated :math:`\hat{f}` to :math:`f`.

        Args:
            scaled_value: A function value from the Taylor expansion of :math:`\hat{f}(x)`.

        Returns:
            The ``scaled_value`` mapped back to the domain of :math:`f`, by first inverting
            the transformation used for the Taylor approximation and then mapping back from
            :math:`[0, 1]` to the original domain.
        