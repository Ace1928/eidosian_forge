import warnings
from typing import Iterable
from functools import lru_cache
import numpy as np
from scipy.linalg import block_diag
import pennylane as qml
from pennylane.operation import (
from pennylane.ops.qubit.matrix_ops import QubitUnitary
from pennylane.ops.qubit.parametric_ops_single_qubit import stack_last
from .controlled import ControlledOp
from .controlled_decompositions import decompose_mcx
class CSWAP(ControlledOp):
    """CSWAP(wires)
    The controlled-swap operator

    .. math:: CSWAP = \\begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\\\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\\\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\\\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\\\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\\\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \\end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    num_wires = 3
    'int : Number of wires that the operation acts on.'
    num_params = 0
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = ()
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    name = 'CSWAP'

    def _flatten(self):
        return (tuple(), (self.wires,))

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(metadata[0])

    def __init__(self, wires, id=None):
        control_wires = wires[:1]
        target_wires = wires[1:]
        super().__init__(qml.SWAP(wires=target_wires), control_wires, id=id)

    def __repr__(self):
        return f'CSWAP(wires={self.wires.tolist()})'

    @staticmethod
    @lru_cache()
    def compute_matrix():
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CSWAP.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.CSWAP.compute_matrix())
        [[1 0 0 0 0 0 0 0]
         [0 1 0 0 0 0 0 0]
         [0 0 1 0 0 0 0 0]
         [0 0 0 1 0 0 0 0]
         [0 0 0 0 1 0 0 0]
         [0 0 0 0 0 0 1 0]
         [0 0 0 0 0 1 0 0]
         [0 0 0 0 0 0 0 1]]
        """
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]])

    @staticmethod
    def compute_decomposition(wires):
        """Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.CSWAP.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.CSWAP.compute_decomposition((0,1,2)))
        [Toffoli(wires=[0, 2, 1]), Toffoli(wires=[0, 1, 2]), Toffoli(wires=[0, 2, 1])]

        """
        decomp_ops = [qml.Toffoli(wires=[wires[0], wires[2], wires[1]]), qml.Toffoli(wires=[wires[0], wires[1], wires[2]]), qml.Toffoli(wires=[wires[0], wires[2], wires[1]])]
        return decomp_ops