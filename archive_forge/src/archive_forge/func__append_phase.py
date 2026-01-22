from __future__ import annotations
import itertools
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.mixins import generate_apidocs, AdjointMixin
from qiskit.circuit import QuantumCircuit, Instruction
from .dihedral_circuits import _append_circuit
from .polynomial import SpecialPolynomial
def _append_phase(self, k, i):
    """Apply an k-th power of T to this element.
        Left multiply the element by T_i^k.
        """
    if not 0 <= i < self.num_qubits:
        raise QiskitError('phase qubit out of bounds.')
    if self.shift[i] == 1:
        k = 7 * k % 8
    support = np.arange(self.num_qubits)[np.nonzero(self.linear[i])]
    subsets_2 = itertools.combinations(support, 2)
    subsets_3 = itertools.combinations(support, 3)
    for j in support:
        value = self.poly.get_term([j])
        self.poly.set_term([j], (value + k) % 8)
    for j in subsets_2:
        value = self.poly.get_term(list(j))
        self.poly.set_term(list(j), (value + -2 * k) % 8)
    for j in subsets_3:
        value = self.poly.get_term(list(j))
        self.poly.set_term(list(j), (value + 4 * k) % 8)