import numpy as np
from pennylane.operation import Operation, AdjointUndefinedError
from pennylane.wires import Wires
from .parametric_ops import validate_subspace
class TSWAP(Operation):
    """TSWAP(wires)
    The ternary swap operator.

    This operation is analogous to the qubit SWAP and acts on two-qutrit computational basis states
    according to :math:`TSWAP\\vert i, j\\rangle = \\vert j, i \\rangle`. Its matrix representation is

    .. math:: TSWAP = \\begin{bmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\\\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\\\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\\\
            \\end{bmatrix}

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    num_wires = 2
    num_params = 0
    'int: Number of trainable parameters that the operator depends on.'

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or 'TSWAP'

    @staticmethod
    def compute_matrix():
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.TSWAP.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.TSWAP.compute_matrix())
        [[1 0 0 0 0 0 0 0 0]
         [0 0 0 1 0 0 0 0 0]
         [0 0 0 0 0 0 1 0 0]
         [0 1 0 0 0 0 0 0 0]
         [0 0 0 0 1 0 0 0 0]
         [0 0 0 0 0 0 0 1 0]
         [0 0 1 0 0 0 0 0 0]
         [0 0 0 0 0 1 0 0 0]
         [0 0 0 0 0 0 0 0 1]]
        """
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    @staticmethod
    def compute_eigvals():
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.
        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.TSWAP.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.TSWAP.compute_eigvals())
        [ 1. -1.  1. -1.  1. -1.  1.  1.  1.]
        """
        return np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0])

    def pow(self, z):
        return super().pow(z % 2)

    def adjoint(self):
        return TSWAP(wires=self.wires)