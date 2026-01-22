import warnings
import numpy as np
import pennylane as qml
from pennylane.operation import Operation
Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.AQFT.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on
            order (int): order of approximation

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> qml.AQFT.compute_decomposition((0, 1, 2), 3, order=1)
        [Hadamard(wires=[0]), ControlledPhaseShift(1.5707963267948966, wires=[1, 0]), Hadamard(wires=[1]), ControlledPhaseShift(1.5707963267948966, wires=[2, 1]), Hadamard(wires=[2]), SWAP(wires=[0, 2])]

        