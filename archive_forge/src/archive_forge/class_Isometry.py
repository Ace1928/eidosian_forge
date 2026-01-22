from __future__ import annotations
import itertools
import numpy as np
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_isometry
from .diagonal import Diagonal
from .uc import UCGate
from .mcg_up_to_diagonal import MCGupDiag
class Isometry(Instruction):
    """Decomposition of arbitrary isometries from :math:`m` to :math:`n` qubits.

    In particular, this allows to decompose unitaries (m=n) and to do state preparation (:math:`m=0`).

    The decomposition is based on [1].

    **References:**

    [1] Iten et al., Quantum circuits for isometries (2016).
        `Phys. Rev. A 93, 032318 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.032318>`__.

    """

    def __init__(self, isometry: np.ndarray, num_ancillas_zero: int, num_ancillas_dirty: int, epsilon: float=_EPS) -> None:
        """
        Args:
            isometry: An isometry from :math:`m` to :math`n` qubits, i.e., a complex
                ``np.ndarray`` of dimension :math:`2^n \\times 2^m` with orthonormal columns (given
                in the computational basis specified by the order of the ancillas
                and the input qubits, where the ancillas are considered to be more
                significant than the input qubits).
            num_ancillas_zero: Number of additional ancillas that start in the state :math:`|0\\rangle`
                (the :math:`n-m` ancillas required for providing the output of the isometry are
                not accounted for here).
            num_ancillas_dirty: Number of additional ancillas that start in an arbitrary state.
            epsilon: Error tolerance of calculations.
        """
        isometry = np.array(isometry, dtype=complex)
        if len(isometry.shape) == 1:
            isometry = isometry.reshape(isometry.shape[0], 1)
        self.iso_data = isometry
        self.num_ancillas_zero = num_ancillas_zero
        self.num_ancillas_dirty = num_ancillas_dirty
        self._inverse = None
        self._epsilon = epsilon
        n = np.log2(isometry.shape[0])
        m = np.log2(isometry.shape[1])
        if not n.is_integer() or n < 0:
            raise QiskitError('The number of rows of the isometry is not a non negative power of 2.')
        if not m.is_integer() or m < 0:
            raise QiskitError('The number of columns of the isometry is not a non negative power of 2.')
        if m > n:
            raise QiskitError("The input matrix has more columns than rows and hence it can't be an isometry.")
        if not is_isometry(isometry, self._epsilon):
            raise QiskitError('The input matrix has non orthonormal columns and hence it is not an isometry.')
        num_qubits = int(n) + num_ancillas_zero + num_ancillas_dirty
        super().__init__('isometry', num_qubits, 0, [isometry])

    def _define(self):
        gate = self.inv_gate()
        gate = gate.inverse()
        q = QuantumRegister(self.num_qubits)
        iso_circuit = QuantumCircuit(q)
        iso_circuit.append(gate, q[:])
        self.definition = iso_circuit

    def inverse(self, annotated: bool=False):
        self.params = []
        inv = super().inverse(annotated=annotated)
        self.params = [self.iso_data]
        return inv

    def _gates_to_uncompute(self):
        """
        Call to create a circuit with gates that take the desired isometry to the first 2^m columns
         of the 2^n*2^n identity matrix (see https://arxiv.org/abs/1501.06911)
        """
        q = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(q)
        q_input, q_ancillas_for_output, q_ancillas_zero, q_ancillas_dirty = self._define_qubit_role(q)
        remaining_isometry = self.iso_data.astype(complex)
        diag = []
        m = int(np.log2(self.iso_data.shape[1]))
        for column_index in range(2 ** m):
            self._decompose_column(circuit, q, diag, remaining_isometry, column_index)
            diag.append(remaining_isometry[column_index, 0])
            remaining_isometry = remaining_isometry[:, 1:]
        if len(diag) > 1 and (not _diag_is_identity_up_to_global_phase(diag, self._epsilon)):
            diagonal = Diagonal(np.conj(diag))
            circuit.append(diagonal, q_input)
        return circuit

    def _decompose_column(self, circuit, q, diag, remaining_isometry, column_index):
        """
        Decomposes the column with index column_index.
        """
        n = int(np.log2(self.iso_data.shape[0]))
        for s in range(n):
            self._disentangle(circuit, q, diag, remaining_isometry, column_index, s)

    def _disentangle(self, circuit, q, diag, remaining_isometry, column_index, s):
        """
        Disentangle the s-th significant qubit (starting with s = 0) into the zero or the one state
        (dependent on column_index)
        """
        k = column_index
        k_prime = 0
        v = remaining_isometry
        n = int(np.log2(self.iso_data.shape[0]))
        index1 = 2 * _a(k, s + 1) * 2 ** s + _b(k, s + 1)
        index2 = (2 * _a(k, s + 1) + 1) * 2 ** s + _b(k, s + 1)
        target_label = n - s - 1
        if _k_s(k, s) == 0 and _b(k, s + 1) != 0 and (np.abs(v[index2, k_prime]) > self._epsilon):
            gate = _reverse_qubit_state([v[index1, k_prime], v[index2, k_prime]], 0, self._epsilon)
            control_labels = [i for i, x in enumerate(_get_binary_rep_as_list(k, n)) if x == 1 and i != target_label]
            diagonal_mcg = self._append_mcg_up_to_diagonal(circuit, q, gate, control_labels, target_label)
            _apply_multi_controlled_gate(v, control_labels, target_label, gate)
            diag_mcg_inverse = np.conj(diagonal_mcg).tolist()
            _apply_diagonal_gate(v, control_labels + [target_label], diag_mcg_inverse)
            _apply_diagonal_gate_to_diag(diag, control_labels + [target_label], diag_mcg_inverse, n)
        single_qubit_gates = self._find_squs_for_disentangling(v, k, s)
        if not _ucg_is_identity_up_to_global_phase(single_qubit_gates, self._epsilon):
            control_labels = list(range(target_label))
            diagonal_ucg = self._append_ucg_up_to_diagonal(circuit, q, single_qubit_gates, control_labels, target_label)
            diagonal_ucg_inverse = np.conj(diagonal_ucg).tolist()
            single_qubit_gates = _merge_UCGate_and_diag(single_qubit_gates, diagonal_ucg_inverse)
            _apply_ucg(v, len(control_labels), single_qubit_gates)
            _apply_diagonal_gate_to_diag(diag, control_labels + [target_label], diagonal_ucg_inverse, n)

    def _find_squs_for_disentangling(self, v, k, s):
        k_prime = 0
        n = int(np.log2(self.iso_data.shape[0]))
        if _b(k, s + 1) == 0:
            i_start = _a(k, s + 1)
        else:
            i_start = _a(k, s + 1) + 1
        id_list = [np.eye(2, 2) for _ in range(i_start)]
        squs = [_reverse_qubit_state([v[2 * i * 2 ** s + _b(k, s), k_prime], v[(2 * i + 1) * 2 ** s + _b(k, s), k_prime]], _k_s(k, s), self._epsilon) for i in range(i_start, 2 ** (n - s - 1))]
        return id_list + squs

    def _append_ucg_up_to_diagonal(self, circ, q, single_qubit_gates, control_labels, target_label):
        q_input, q_ancillas_for_output, q_ancillas_zero, q_ancillas_dirty = self._define_qubit_role(q)
        n = int(np.log2(self.iso_data.shape[0]))
        qubits = q_input + q_ancillas_for_output
        control_qubits = _reverse_qubit_oder(_get_qubits_by_label(control_labels, qubits, n))
        target_qubit = _get_qubits_by_label([target_label], qubits, n)[0]
        ucg = UCGate(single_qubit_gates, up_to_diagonal=True)
        circ.append(ucg, [target_qubit] + control_qubits)
        return ucg._get_diagonal()

    def _append_mcg_up_to_diagonal(self, circ, q, gate, control_labels, target_label):
        q_input, q_ancillas_for_output, q_ancillas_zero, q_ancillas_dirty = self._define_qubit_role(q)
        n = int(np.log2(self.iso_data.shape[0]))
        qubits = q_input + q_ancillas_for_output
        control_qubits = _reverse_qubit_oder(_get_qubits_by_label(control_labels, qubits, n))
        target_qubit = _get_qubits_by_label([target_label], qubits, n)[0]
        ancilla_dirty_labels = [i for i in range(n) if i not in control_labels + [target_label]]
        ancillas_dirty = _reverse_qubit_oder(_get_qubits_by_label(ancilla_dirty_labels, qubits, n)) + q_ancillas_dirty
        mcg_up_to_diag = MCGupDiag(gate, len(control_qubits), len(q_ancillas_zero), len(ancillas_dirty))
        circ.append(mcg_up_to_diag, [target_qubit] + control_qubits + q_ancillas_zero + ancillas_dirty)
        return mcg_up_to_diag._get_diagonal()

    def _define_qubit_role(self, q):
        n = int(np.log2(self.iso_data.shape[0]))
        m = int(np.log2(self.iso_data.shape[1]))
        q_input = q[:m]
        q_ancillas_for_output = q[m:n]
        q_ancillas_zero = q[n:n + self.num_ancillas_zero]
        q_ancillas_dirty = q[n + self.num_ancillas_zero:]
        return (q_input, q_ancillas_for_output, q_ancillas_zero, q_ancillas_dirty)

    def validate_parameter(self, parameter):
        """Isometry parameter has to be an ndarray."""
        if isinstance(parameter, np.ndarray):
            return parameter
        if isinstance(parameter, (list, int)):
            return parameter
        else:
            raise CircuitError(f'invalid param type {type(parameter)} for gate {self.name}')

    def inv_gate(self):
        """Return the adjoint of the unitary."""
        if self._inverse is None:
            iso_circuit = self._gates_to_uncompute()
            self._inverse = iso_circuit.to_instruction()
        return self._inverse