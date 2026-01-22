import inspect
import warnings
import numpy as np
import pennylane as qml
from pennylane.ops import Hamiltonian, SProd, Prod, Sum
def _generator_hamiltonian(gen, op):
    """Return the generator as type :class:`~.Hamiltonian`."""
    wires = op.wires
    if isinstance(gen, qml.Hamiltonian):
        H = gen
    elif isinstance(gen, (qml.Hermitian, qml.SparseHamiltonian)):
        if isinstance(gen, qml.Hermitian):
            mat = gen.parameters[0]
        elif isinstance(gen, qml.SparseHamiltonian):
            mat = gen.parameters[0].toarray()
        H = qml.pauli_decompose(mat, wire_order=wires, hide_identity=True)
    elif isinstance(gen, qml.operation.Observable):
        H = 1.0 * gen
    return H