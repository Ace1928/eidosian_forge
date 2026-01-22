import numpy as np
from pennylane.operation import Operation, AdjointUndefinedError
from pennylane.wires import Wires
from .parametric_ops import validate_subspace
class TAdd(Operation):
    """TAdd(wires)
    The 2-qutrit controlled add gate

    The construction of this operator is based on definition 7 from
    `Yeh et al. (2022) <https://arxiv.org/abs/2204.00552>`_.
    It performs the controlled :class:`~.TShift` operation, and sends
    :math:`\\hbox{TAdd} \\vert i \\rangle \\vert j \\rangle = \\vert i \\rangle \\vert i + j \\rangle`,
    where addition is taken modulo 3. The matrix representation is

    .. math:: TAdd = \\begin{bmatrix}
                        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
                        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
                        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
                        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\\\
                        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\\\
                        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\\\
                        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\\\
                        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\\\
                        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0
                    \\end{bmatrix}

    .. note:: The first wire provided corresponds to the **control qutrit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    num_wires = 2
    'int: Number of wires that the operator acts on.'
    num_params = 0
    'int: Number of trainable parameters that the operator depends on.'

    @staticmethod
    def compute_matrix():
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.TAdd.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.TAdd.compute_matrix())
        [[1 0 0 0 0 0 0 0 0]
         [0 1 0 0 0 0 0 0 0]
         [0 0 1 0 0 0 0 0 0]
         [0 0 0 0 0 1 0 0 0]
         [0 0 0 1 0 0 0 0 0]
         [0 0 0 0 1 0 0 0 0]
         [0 0 0 0 0 0 0 1 0]
         [0 0 0 0 0 0 0 0 1]
         [0 0 0 0 0 0 1 0 0]]
        """
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0, 0]])

    @staticmethod
    def compute_eigvals():
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.
        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.TAdd.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.TAdd.compute_eigvals())
        [-0.5+0.8660254j -0.5-0.8660254j  1. +0.j        -0.5+0.8660254j -0.5-0.8660254j  1. +0.j         1. +0.j         1. +0.j         1. +0.j       ]
        """
        return np.array([OMEGA, OMEGA ** 2, 1, OMEGA, OMEGA ** 2, 1, 1, 1, 1])

    def pow(self, z):
        return super().pow(z % 3)

    @property
    def control_wires(self):
        return Wires(self.wires[0])