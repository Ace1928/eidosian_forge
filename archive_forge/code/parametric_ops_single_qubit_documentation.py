import functools
import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from .non_parametric_ops import Hadamard, PauliX, PauliY, PauliZ
Simplifies into :class:`~.RX`, :class:`~.RY`, or :class:`~.PhaseShift` gates
        if possible.

        >>> qml.U3(0.1, 0, 0, wires=0).simplify()
        RY(0.1, wires=[0])

        