from copy import copy
from collections.abc import Sequence
import numpy as np
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane.operation import AnyWires, Observable, Operation
from pennylane.wires import Wires
from .matrix_ops import QubitUnitary
Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.StateVectorProjector.diagonalizing_gates`.

        Args:
            state_vector (Iterable): state vector that the operator projects on.
            wires (Iterable[Any], Wires): wires that the operator acts on.
        Returns:
            list[.Operator]: list of diagonalizing gates.

        **Example**

        >>> state_vector = np.array([1., 1j])/np.sqrt(2)
        >>> StateVectorProjector.compute_diagonalizing_gates(state_vector, wires=[0])
        [QubitUnitary(array([[ 0.70710678+0.j        ,  0.        -0.70710678j],
                             [ 0.        +0.70710678j, -0.70710678+0.j        ]]), wires=[0])]
        