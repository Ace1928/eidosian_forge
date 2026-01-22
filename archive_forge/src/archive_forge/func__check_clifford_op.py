import math
import warnings
from itertools import product
from typing import Sequence, Callable
import pennylane as qml
from pennylane.ops import Adjoint
from pennylane.queuing import QueuingManager
from pennylane.transforms.core import transform
from pennylane.tape import QuantumTape
from pennylane.transforms.optimization import (
from pennylane.transforms.optimization.optimization_utils import find_next_gate, _fuse_global_phases
from pennylane.ops.op_math.decompositions.solovay_kitaev import sk_decomposition
def _check_clifford_op(op, use_decomposition=False):
    """Checks if an operator is Clifford or not.

    For a given unitary operator :math:`U` acting on :math:`N` qubits, this method checks that the
    transformation :math:`UPU^{\\dagger}` maps the Pauli tensor products :math:`P = {I, X, Y, Z}^{\\otimes N}`
    to Pauli tensor products using the decomposition of the matrix for :math:`U` in the Pauli basis.

    Args:
        op (~pennylane.operation.Operation): the operator that needs to be tested
        use_decomposition (bool): if ``True``, use operator's decomposition to compute the matrix, in case
            it doesn't define a ``compute_matrix`` method. Default is ``False``.

    Returns:
        Bool that represents whether the provided operator is Clifford or not.
    """
    if not op.has_matrix and (not use_decomposition) or (use_decomposition and (not op.expand().wires)):
        return False
    pauli_terms = qml.pauli_decompose(qml.matrix(op), wire_order=op.wires, check_hermitian=False)
    pauli_terms_adj = qml.Hamiltonian(qml.math.conj(pauli_terms.coeffs), pauli_terms.ops)

    def pauli_group(x):
        return [qml.Identity(x), qml.X(x), qml.Y(x), qml.Z(x)]
    pauli_sens = [qml.pauli.pauli_sentence(qml.prod(*pauli)) for pauli in product(*(pauli_group(idx) for idx in op.wires))]
    pauli_hams = (pauli_sen.hamiltonian(wire_order=op.wires) for pauli_sen in pauli_sens)
    for pauli_prod in product([pauli_terms], pauli_hams, [pauli_terms_adj]):
        upu = qml.pauli.pauli_sentence(qml.prod(*pauli_prod))
        upu.simplify()
        if len(upu) != 1:
            return False
    return True