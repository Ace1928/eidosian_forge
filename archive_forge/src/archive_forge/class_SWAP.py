import cmath
from copy import copy
from functools import lru_cache
import numpy as np
from scipy import sparse
import pennylane as qml
from pennylane.operation import Observable, Operation
from pennylane.utils import pauli_eigs
from pennylane.wires import Wires
class SWAP(Operation):
    """SWAP(wires)
    The swap operator

    .. math:: SWAP = \\begin{bmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 0 & 1 & 0\\\\
            0 & 1 & 0 & 0\\\\
            0 & 0 & 0 & 1
        \\end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    num_wires = 2
    num_params = 0
    'int: Number of trainable parameters that the operator depends on.'
    batch_size = None

    @staticmethod
    @lru_cache()
    def compute_matrix():
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SWAP.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.SWAP.compute_matrix())
        [[1 0 0 0]
         [0 0 1 0]
         [0 1 0 0]
         [0 0 0 1]]
        """
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    @staticmethod
    def compute_decomposition(wires):
        """Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.SWAP.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.SWAP.compute_decomposition((0,1)))
        [CNOT(wires=[0, 1]), CNOT(wires=[1, 0]), CNOT(wires=[0, 1])]

        """
        return [qml.CNOT(wires=[wires[0], wires[1]]), qml.CNOT(wires=[wires[1], wires[0]]), qml.CNOT(wires=[wires[0], wires[1]])]

    def pow(self, z):
        return super().pow(z % 2)

    def adjoint(self):
        return SWAP(wires=self.wires)

    def _controlled(self, wire):
        return qml.CSWAP(wires=wire + self.wires)

    @property
    def is_hermitian(self):
        return True