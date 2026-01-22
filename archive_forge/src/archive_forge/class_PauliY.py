import cmath
from copy import copy
from functools import lru_cache
import numpy as np
from scipy import sparse
import pennylane as qml
from pennylane.operation import Observable, Operation
from pennylane.utils import pauli_eigs
from pennylane.wires import Wires
class PauliY(Observable, Operation):
    """
    The Pauli Y operator

    .. math:: \\sigma_y = \\begin{bmatrix} 0 & -i \\\\ i & 0\\end{bmatrix}.

    .. seealso:: The equivalent short-form alias :class:`~Y`

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
    basis = 'Y'
    batch_size = None
    _queue_category = '_ops'

    def __init__(self, *params, wires=None, id=None):
        super().__init__(*params, wires=wires, id=id)
        self._pauli_rep = qml.pauli.PauliSentence({qml.pauli.PauliWord({self.wires[0]: 'Y'}): 1.0})

    def __repr__(self):
        """String representation."""
        wire = self.wires[0]
        if isinstance(wire, str):
            return f"Y('{wire}')"
        return f'Y({wire})'

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or 'Y'

    @property
    def name(self):
        return 'PauliY'

    @staticmethod
    @lru_cache()
    def compute_matrix():
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Y.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.Y.compute_matrix())
        [[ 0.+0.j -0.-1.j]
         [ 0.+1.j  0.+0.j]]
        """
        return np.array([[0, -1j], [1j, 0]])

    @staticmethod
    @lru_cache()
    def compute_sparse_matrix():
        return sparse.csr_matrix([[0, -1j], [1j, 0]])

    @staticmethod
    def compute_eigvals():
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.Y.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.Y.compute_eigvals())
        [ 1 -1]
        """
        return pauli_eigs(1)

    @staticmethod
    def compute_diagonalizing_gates(wires):
        """Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \\Sigma U^{\\dagger}` where
        :math:`\\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.Y.diagonalizing_gates`.

        Args:
            wires (Iterable[Any], Wires): wires that the operator acts on
        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> print(qml.Y.compute_diagonalizing_gates(wires=[0]))
        [Z(0), S(wires=[0]), Hadamard(wires=[0])]
        """
        return [Z(wires=wires), S(wires=wires), Hadamard(wires=wires)]

    @staticmethod
    def compute_decomposition(wires):
        """Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \\dots O_n.

        .. seealso:: :meth:`~.Y.decomposition`.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.Y.compute_decomposition(0))
        [PhaseShift(1.5707963267948966, wires=[0]),
        RY(3.141592653589793, wires=[0]),
        PhaseShift(1.5707963267948966, wires=[0])]

        """
        return [qml.PhaseShift(np.pi / 2, wires=wires), qml.RY(np.pi, wires=wires), qml.PhaseShift(np.pi / 2, wires=wires)]

    def adjoint(self):
        return Y(wires=self.wires)

    def pow(self, z):
        return super().pow(z % 2)

    def _controlled(self, wire):
        return qml.CY(wires=Wires(wire) + self.wires)

    def single_qubit_rot_angles(self):
        return [0.0, np.pi, 0.0]