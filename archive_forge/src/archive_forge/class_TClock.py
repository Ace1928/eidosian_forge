import numpy as np
from pennylane.operation import Operation, AdjointUndefinedError
from pennylane.wires import Wires
from .parametric_ops import validate_subspace
class TClock(Operation):
    """TClock(wires)
    Ternary Clock gate

    The construction of this operator is based on equation 1 from
    `Yeh et al. (2022) <https://arxiv.org/abs/2204.00552>`_.

    .. math:: TClock = \\begin{bmatrix}
                        1 & 0      & 0        \\\\
                        0 & \\omega & 0        \\\\
                        0 & 0      & \\omega^2
                    \\end{bmatrix}

    where :math:`\\omega = e^{2 \\pi i / 3}`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    'int: Number of wires that the operator acts on.'
    num_params = 0
    'int: Number of trainable parameters that the operator depends on.'

    @staticmethod
    def compute_matrix():
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.TClock.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.TClock.compute_matrix())
        [[ 1. +0.j         0. +0.j         0. +0.j       ]
         [ 0. +0.j        -0.5+0.8660254j  0. +0.j       ]
         [ 0. +0.j         0. +0.j        -0.5-0.8660254j]]
        """
        return np.diag([1, OMEGA, OMEGA ** 2])

    @staticmethod
    def compute_eigvals():
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.
        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.TClock.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.TClock.compute_eigvals())
        [ 1. +0.j        -0.5+0.8660254j -0.5-0.8660254j]
        """
        return np.array([1, OMEGA, OMEGA ** 2])

    def pow(self, z):
        return super().pow(z % 3)