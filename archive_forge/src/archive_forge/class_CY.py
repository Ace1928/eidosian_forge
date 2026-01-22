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
class CY(ControlledOp):
    """CY(wires)
    The controlled-Y operator

    .. math:: CY = \\begin{bmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0\\\\
            0 & 0 & 0 & -i\\\\
            0 & 0 & i & 0
        \\end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.
    """
    num_wires = 2
    'int: Number of wires that the operator acts on.'
    num_params = 0
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = ()
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    name = 'CY'

    def _flatten(self):
        return (tuple(), (self.wires,))

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(metadata[0])

    def __init__(self, wires, id=None):
        control_wire, wire = wires
        super().__init__(qml.Y(wire), control_wire, id=id)

    def __repr__(self):
        return f'CY(wires={self.wires.tolist()})'

    @staticmethod
    @lru_cache()
    def compute_matrix():
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CY.matrix`


        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.CY.compute_matrix())
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j -0.-1.j]
         [ 0.+0.j  0.+0.j  0.+1.j  0.+0.j]]
        """
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])

    @staticmethod
    def compute_decomposition(wires):
        """Representation of the operator as a product of other operators (static method).


        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.CY.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.CY.compute_decomposition([0, 1]))
        [CRY(3.141592653589793, wires=[0, 1])), S(wires=[0])]

        """
        return [qml.CRY(np.pi, wires=wires), qml.S(wires=wires[0])]