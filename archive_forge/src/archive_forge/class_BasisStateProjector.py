from copy import copy
from collections.abc import Sequence
import numpy as np
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane.operation import AnyWires, Observable, Operation
from pennylane.wires import Wires
from .matrix_ops import QubitUnitary
class BasisStateProjector(Projector, Operation):
    """Observable corresponding to the state projector :math:`P=\\ket{\\phi}\\bra{\\phi}`, where
    :math:`\\phi` denotes a basis state."""

    def __init__(self, state, wires, id=None):
        wires = Wires(wires)
        state = list(qml.math.toarray(state).astype(int))
        if not set(state).issubset({0, 1}):
            raise ValueError(f'Basis state must only consist of 0s and 1s; got {state}')
        super().__init__(state, wires=wires, id=id)

    def __new__(cls, *_, **__):
        return object.__new__(cls)

    def label(self, decimals=None, base_label=None, cache=None):
        """A customizable string representation of the operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label.
            cache=None (dict): dictionary that caries information between label calls
                in the same drawing.

        Returns:
            str: label to use in drawings.

        **Example:**

        >>> BasisStateProjector([0, 1, 0], wires=(0, 1, 2)).label()
        '|010⟩⟨010|'

        """
        if base_label is not None:
            return base_label
        basis_string = ''.join((str(int(i)) for i in self.parameters[0]))
        return f'|{basis_string}⟩⟨{basis_string}|'

    @staticmethod
    def compute_matrix(basis_state):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.BasisStateProjector.matrix`

        Args:
            basis_state (Iterable): basis state to project on

        Returns:
            ndarray: matrix

        **Example**

        >>> BasisStateProjector.compute_matrix([0, 1])
        [[0. 0. 0. 0.]
         [0. 1. 0. 0.]
         [0. 0. 0. 0.]
         [0. 0. 0. 0.]]
        """
        m = np.zeros((2 ** len(basis_state), 2 ** len(basis_state)))
        idx = int(''.join((str(i) for i in basis_state)), 2)
        m[idx, idx] = 1
        return m

    @staticmethod
    def compute_eigvals(basis_state):
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.BasisStateProjector.eigvals`

        Args:
            basis_state (Iterable): basis state to project on

        Returns:
            array: eigenvalues

        **Example**

        >>> BasisStateProjector.compute_eigvals([0, 1])
        [0. 1. 0. 0.]
        """
        w = np.zeros(2 ** len(basis_state))
        idx = int(''.join((str(i) for i in basis_state)), 2)
        w[idx] = 1
        return w

    @staticmethod
    def compute_diagonalizing_gates(basis_state, wires):
        """Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \\Sigma U^{\\dagger}` where
        :math:`\\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.BasisStateProjector.diagonalizing_gates`.

        Args:
            basis_state (Iterable): basis state that the operator projects on
            wires (Iterable[Any], Wires): wires that the operator acts on
        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> BasisStateProjector.compute_diagonalizing_gates([0, 1, 0, 0], wires=[0, 1])
        []
        """
        return []