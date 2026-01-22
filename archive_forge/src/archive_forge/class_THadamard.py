import numpy as np
from pennylane.operation import Operation, AdjointUndefinedError
from pennylane.wires import Wires
from .parametric_ops import validate_subspace
class THadamard(Operation):
    """THadamard(wires, subspace)
    The ternary Hadamard operator

    Performs the Hadamard operation on a 2D subspace, if specified. The subspace is
    given as a keyword argument and determines which two of three single-qutrit basis states the
    operation applies to. When a subspace is not specified, the generalized Hadamard operation
    is used.

    The construction of this operator is based on section 2 of
    `Di et al. (2012) <https://arxiv.org/abs/1105.5485>`_ when the subspace is specified, and
    definition 4 and equation 5 from `Yeh et al. (2022) <https://arxiv.org/abs/2204.00552>`_
    when no subspace is specified. The operator definition of the ``subspace=None`` case is

    .. math:: \\text{THadamard} = \\frac{-i}{\\sqrt{3}}\\begin{bmatrix}
                    1 & 1 & 1 \\\\
                    1 & \\omega & \\omega^2 \\\\
                    1 & \\omega^2 & \\omega \\\\
                \\end{bmatrix}

    where :math:`\\omega = \\exp(2 \\pi i / 3)`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
        subspace (Optional[Sequence[int]]): the 2D subspace on which to apply the operation.
            This should be `None` for the generalized Hadamard.

    **Example**

    The specified subspace will determine which basis states the operation actually
    applies to:

    >>> qml.THadamard(wires=0, subspace=(0, 1)).matrix()
    array([[ 0.70710678+0.j,  0.70710678+0.j,  0.        +0.j],
           [ 0.70710678+0.j, -0.70710678+0.j,  0.        +0.j],
           [ 0.        +0.j,  0.        +0.j,  1.        +0.j]])

    >>> qml.THadamard(wires=0, subspace=(0, 2)).matrix()
    array([[ 0.70710678+0.j,  0.        +0.j,  0.70710678+0.j],
           [ 0.        +0.j,  1.        +0.j,  0.        +0.j],
           [ 0.70710678+0.j,  0.        +0.j, -0.70710678+0.j]])

    >>> qml.THadamard(wires=0, subspace=(1, 2)).matrix()
    array([[ 1.        +0.j,  0.        +0.j,  0.        +0.j],
           [ 0.        +0.j,  0.70710678+0.j,  0.70710678+0.j],
           [ 0.        +0.j,  0.70710678+0.j, -0.70710678+0.j]])

    >>> qml.THadamard(wires=0, subspace=None).matrix()
    array([[ 0. -0.57735027j,  0. -0.57735027j,  0. -0.57735027j],
           [ 0. -0.57735027j,  0.5+0.28867513j, -0.5+0.28867513j],
           [ 0. -0.57735027j, -0.5+0.28867513j,  0.5+0.28867513j]])
    """
    num_wires = 1
    num_params = 0
    'int: Number of trainable parameters that the operator depends on.'

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or 'TH'

    def __init__(self, wires, subspace=None):
        self._subspace = validate_subspace(subspace) if subspace is not None else None
        self._hyperparameters = {'subspace': self.subspace}
        super().__init__(wires=wires)

    @property
    def subspace(self):
        """The single-qutrit basis states which the operator acts on

        This property returns the 2D subspace on which the operator acts if specified,
        or None if no subspace is defined. This subspace determines which two single-qutrit
        basis states the operator acts on. The remaining basis state is not affected by the
        operator.

        Returns:
            tuple[int] or None: subspace on which operator acts, if specified, else None
        """
        return self._subspace

    @staticmethod
    def compute_matrix(subspace=None):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.THadamard.matrix`

        Args:
            subspace (Sequence[int]): the 2D subspace on which to apply operation. This should be
            `None` for the generalized Hadamard.

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.THadamard.compute_matrix(subspace=(0, 2)))
        array([[ 0.70710678+0.j,  0.        +0.j,  0.70710678+0.j],
               [ 0.        +0.j,  1.        +0.j,  0.        +0.j],
               [ 0.70710678+0.j,  0.        +0.j, -0.70710678+0.j]])
        """
        if subspace is None:
            return -1j / np.sqrt(3) * np.array([[1, 1, 1], [1, OMEGA, OMEGA ** 2], [1, OMEGA ** 2, OMEGA]])
        mat = np.eye(3, dtype=np.complex128)
        unused_ind = list({0, 1, 2}.difference(set(subspace))).pop()
        mat[unused_ind, unused_ind] = np.sqrt(2)
        mat[subspace[0], subspace[1]] = 1
        mat[subspace[1], subspace[0]] = 1
        mat[subspace[1], subspace[1]] = -1
        return mat / np.sqrt(2)

    @property
    def has_adjoint(self):
        return self.subspace is not None

    def adjoint(self):
        if self.subspace is None:
            raise AdjointUndefinedError
        return THadamard(wires=self.wires, subspace=self.subspace)

    def pow(self, z):
        new_exp = z % 4 if self.subspace is None else z % 2
        return super().pow(new_exp)