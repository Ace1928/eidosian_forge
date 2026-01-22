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
def _append_ucg_up_to_diagonal(self, circ, q, single_qubit_gates, control_labels, target_label):
    q_input, q_ancillas_for_output, q_ancillas_zero, q_ancillas_dirty = self._define_qubit_role(q)
    n = int(np.log2(self.iso_data.shape[0]))
    qubits = q_input + q_ancillas_for_output
    control_qubits = _reverse_qubit_oder(_get_qubits_by_label(control_labels, qubits, n))
    target_qubit = _get_qubits_by_label([target_label], qubits, n)[0]
    ucg = UCGate(single_qubit_gates, up_to_diagonal=True)
    circ.append(ucg, [target_qubit] + control_qubits)
    return ucg._get_diagonal()