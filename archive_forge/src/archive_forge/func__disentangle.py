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