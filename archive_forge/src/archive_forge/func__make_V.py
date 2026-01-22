import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops import QubitUnitary
def _make_V(dim):
    """Calculates the :math:`\\mathcal{V}` unitary which performs a reflection along the
    :math:`|1\\rangle` state of the end ancilla qubit.

    Args:
        dim (int): dimension of :math:`\\mathcal{V}`

    Returns:
        array: the :math:`\\mathcal{V}` unitary
    """
    assert dim % 2 == 0, 'dimension for _make_V() must be even'
    one = np.array([[0, 0], [0, 1]])
    dim_without_qubit = int(dim / 2)
    return 2 * np.kron(np.eye(dim_without_qubit), one) - np.eye(dim)