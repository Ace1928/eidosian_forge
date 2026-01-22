import functools
import numpy as np
import pennylane as qml
from pennylane.operation import Operation
class TRZ(Operation):
    """The single qutrit Z rotation

    Performs the RZ operation on the specified 2D subspace. The subspace is
    given as a keyword argument and determines which two of three single-qutrit
    basis states the operation applies to.

    The construction of this operator is based on section 3 of
    `Di et al. (2012) <https://arxiv.org/abs/1105.5485>`_.

    .. math:: TRZ^{jk}(\\phi) = \\exp(-i\\phi\\sigma_z^{jk}/2),

    where :math:`\\sigma_z^{jk} = |j\\rangle\\langle j| - |k\\rangle\\langle k|;`
    :math:`j, k \\in \\{0, 1, 2\\}, j < k`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)

    Args:
        phi (float): rotation angle :math:`\\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        subspace (Sequence[int]): the 2D subspace on which to apply operation
        id (str or None): String representing the operation (optional)

    **Example**

    The specified subspace will determine which basis states the operation actually
    applies to:

    >>> qml.TRZ(0.5, wires=0, subspace=(0, 1)).matrix()
    array([[0.96891242-0.24740396j, 0.        +0.j        , 0.        +0.j        ],
           [0.        +0.j        , 0.96891242+0.24740396j, 0.        +0.j        ],
           [0.        +0.j        , 0.        +0.j        , 1.        +0.j        ]])

    >>> qml.TRZ(0.5, wires=0, subspace=(0, 2)).matrix()
    array([[0.96891242-0.24740396j, 0.        +0.j        , 0.        +0.j        ],
           [0.        +0.j        , 1.        +0.j        , 0.        +0.j        ],
           [0.        +0.j        , 0.        +0.j        , 0.96891242+0.24740396j]])

    >>> qml.TRZ(0.5, wires=0, subspace=(1, 2)).matrix()
    array([[1.        +0.j        , 0.        +0.j        , 0.        +0.j        ],
           [0.        +0.j        , 0.96891242-0.24740396j, 0.        +0.j        ],
           [0.        +0.j        , 0.        +0.j        , 0.96891242+0.24740396j]])
    """
    num_wires = 1
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = 'A'
    parameter_frequencies = [(0.5, 1)]

    def generator(self):
        if self.subspace == (0, 1):
            return qml.s_prod(-0.5, qml.GellMann(wires=self.wires, index=3))
        if self.subspace == (0, 2):
            coeffs = [-0.25, -0.25 * np.sqrt(3)]
            obs = [qml.GellMann(wires=self.wires, index=3), qml.GellMann(wires=self.wires, index=8)]
            return qml.dot(coeffs, obs)
        coeffs = [0.25, -0.25 * np.sqrt(3)]
        obs = [qml.GellMann(wires=self.wires, index=3), qml.GellMann(wires=self.wires, index=8)]
        return qml.dot(coeffs, obs)

    def __init__(self, phi, wires, subspace=(0, 1), id=None):
        self._subspace = validate_subspace(subspace)
        self._hyperparameters = {'subspace': self._subspace}
        super().__init__(phi, wires=wires, id=id)

    @property
    def subspace(self):
        """The single-qutrit basis states which the operator acts on

        This subspace determines which two single-qutrit basis states the operator acts on.
        The remaining basis state is not affected by the operator.

        Returns:
            tuple[int]: subspace on which operator acts
        """
        return self._subspace

    @staticmethod
    def compute_matrix(theta, subspace=(0, 1)):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.TRZ.matrix`

        Args:
            theta (tensor_like or float): rotation angle
            subspace (Sequence[int]): the 2D subspace on which to apply operation

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.TRZ.compute_matrix(torch.tensor(0.5), subspace=(0, 2))
        tensor([[0.9689-0.2474j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 1.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.9689+0.2474j]])
        """
        if qml.math.get_interface(theta) == 'tensorflow':
            theta = qml.math.cast_like(theta, 1j)
        p = qml.math.exp(-1j * theta / 2)
        one = qml.math.ones_like(p)
        z = qml.math.zeros_like(p)
        diags = [one, one, one]
        diags[subspace[0]] = p
        diags[subspace[1]] = qml.math.conj(p)
        return qml.math.stack([stack_last([diags[0], z, z]), stack_last([z, diags[1], z]), stack_last([z, z, diags[2]])], axis=-2)

    def adjoint(self):
        return TRZ(-self.data[0], wires=self.wires, subspace=self.subspace)

    def pow(self, z):
        return [TRZ(self.data[0] * z, wires=self.wires, subspace=self.subspace)]