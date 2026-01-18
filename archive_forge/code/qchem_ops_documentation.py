from quantum chemistry applications.
import functools
import numpy as np
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane.operation import Operation
Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.FermionicSWAP.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.FermionicSWAP.compute_decomposition(0.2, wires=(0, 1))
        [Hadamard(wires=[0]),
         Hadamard(wires=[1]),
         MultiRZ(0.1, wires=[0, 1]),
         Hadamard(wires=[0]),
         Hadamard(wires=[1]),
         RX(1.5707963267948966, wires=[0]),
         RX(1.5707963267948966, wires=[1]),
         MultiRZ(0.1, wires=[0, 1]),
         RX(-1.5707963267948966, wires=[0]),
         RX(-1.5707963267948966, wires=[1]),
         RZ(0.1, wires=[0]),
         RZ(0.1, wires=[1]),
         Exp(0.1j Identity)]
        