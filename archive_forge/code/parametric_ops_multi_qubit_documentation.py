import functools
from operator import matmul
import numpy as np
import pennylane as qml
from pennylane.math import expand_matrix
from pennylane.operation import AnyWires, Operation
from pennylane.utils import pauli_eigs
from pennylane.wires import Wires
from .non_parametric_ops import Hadamard, PauliX, PauliY, PauliZ
from .parametric_ops_single_qubit import _can_replace, stack_last, RX, RY, RZ, PhaseShift
Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.
        .. seealso:: :meth:`~.CPhaseShift10.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.CPhaseShift10.compute_decomposition(1.234, wires=(0,1))
        [X(1),
        PhaseShift(0.617, wires=[0]),
        PhaseShift(0.617, wires=[1]),
        CNOT(wires=[0, 1]),
        PhaseShift(-0.617, wires=[1]),
        CNOT(wires=[0, 1]),
        X(1)]

        