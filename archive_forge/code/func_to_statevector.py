from __future__ import annotations
import copy
from numbers import Number
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.operators.mixins.tolerances import TolerancesMixin
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import Pauli, SparsePauliOp
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit._accelerate.pauli_expval import density_expval_pauli_no_x, density_expval_pauli_with_x
from qiskit.quantum_info.states.statevector import Statevector
def to_statevector(self, atol: float | None=None, rtol: float | None=None) -> Statevector:
    """Return a statevector from a pure density matrix.

        Args:
            atol (float): Absolute tolerance for checking operation validity.
            rtol (float): Relative tolerance for checking operation validity.

        Returns:
            Statevector: The pure density matrix's corresponding statevector.
                Corresponds to the eigenvector of the only non-zero eigenvalue.

        Raises:
            QiskitError: if the state is not pure.
        """
    if atol is None:
        atol = self.atol
    if rtol is None:
        rtol = self.rtol
    if not is_hermitian_matrix(self._data, atol=atol, rtol=rtol):
        raise QiskitError('Not a valid density matrix (non-hermitian).')
    evals, evecs = np.linalg.eig(self._data)
    nonzero_evals = evals[abs(evals) > atol]
    if len(nonzero_evals) != 1 or not np.isclose(nonzero_evals[0], 1, atol=atol, rtol=rtol):
        raise QiskitError('Density matrix is not a pure state')
    psi = evecs[:, np.argmax(evals)]
    return Statevector(psi)