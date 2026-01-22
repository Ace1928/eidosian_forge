import functools
import itertools
import numpy as np
import scipy
import pennylane as qml
from pennylane.operation import active_new_opmath
from pennylane.pauli import PauliSentence, PauliWord, pauli_sentence, simplify
from pennylane.pauli.utils import _binary_matrix_from_pws
from pennylane.wires import Wires
def _build_generator(operation, wire_order, op_gen=None):
    """Computes the generator `G` for the general unitary operation :math:`U(\\theta)=e^{iG\\theta}`, where :math:`\\theta` could either be a variational parameter,
    or a constant with some arbitrary fixed value.
    Args:
        operation (Operation): qubit operation to be tapered
        wire_order (Sequence[Any]): order of the wires in the quantum circuit
        op_gen (Hamiltonian): generator of the operation in case it cannot be computed internally.
    Returns:
        Hamiltonian: the generator of the operation
    Raises:
        NotImplementedError: generator of the operation cannot be constructed internally
        ValueError: optional argument `op_gen` is either not a :class:`~.pennylane.Hamiltonian` or a valid generator of the operation
    **Example**
    >>> _build_generator(qml.SingleExcitation, [0, 1], op_wires=[0, 2])
      (-0.25) [Y0 X1]
    + (0.25) [X0 Y1]
    """
    if op_gen is None:
        if operation.num_params < 1:
            gen_mat = 1j * scipy.linalg.logm(qml.matrix(operation, wire_order=wire_order))
            op_gen = qml.pauli_decompose(gen_mat, wire_order=wire_order, hide_identity=True)
            op_gen = qml.simplify(op_gen)
            if op_gen.ops[0].label() == qml.Identity(wires=[wire_order[0]]).label():
                op_gen -= qml.Hamiltonian([op_gen.coeffs[0]], [qml.Identity(wires=wire_order[0])])
        else:
            try:
                op_gen = qml.generator(operation, 'hamiltonian')
            except ValueError as exc:
                raise NotImplementedError(f"Generator for {operation} is not implemented, please provide it with 'op_gen' args.") from exc
    else:
        if not isinstance(op_gen, qml.Hamiltonian):
            raise ValueError(f'Generator for the operation needs to be a qml.Hamiltonian, but got {type(op_gen)}.')
        coeffs = 1.0
        if operation.parameters and isinstance(operation.parameters[0], (float, complex)):
            coeffs = functools.reduce(lambda i, j: i * j, operation.parameters)
        mat1 = scipy.linalg.expm(1j * qml.matrix(op_gen, wire_order=wire_order) * coeffs)
        mat2 = qml.matrix(operation, wire_order=wire_order)
        phase = np.divide(mat1, mat2, out=np.zeros_like(mat1, dtype=complex), where=mat1 != 0)[np.nonzero(np.round(mat1, 10))]
        if not np.allclose(phase / phase[0], np.ones(len(phase))):
            raise ValueError(f"Given op_gen: {op_gen} doesn't seem to be the correct generator for the {operation}.")
    return op_gen