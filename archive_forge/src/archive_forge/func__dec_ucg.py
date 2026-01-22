from __future__ import annotations
import cmath
import math
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
from qiskit.synthesis.one_qubit.one_qubit_decompose import OneQubitEulerDecomposer
from .diagonal import Diagonal
def _dec_ucg(self):
    """
        Call to create a circuit that implements the uniformly controlled gate. If
        up_to_diagonal=True, the circuit implements the gate up to a diagonal gate and
        the diagonal gate is also returned.
        """
    diag = np.ones(2 ** self.num_qubits).tolist()
    q = QuantumRegister(self.num_qubits)
    q_controls = q[1:]
    q_target = q[0]
    circuit = QuantumCircuit(q)
    if not q_controls:
        circuit.unitary(self.params[0], [q])
        return (circuit, diag)
    single_qubit_gates, diag = self._dec_ucg_help()
    for i, gate in enumerate(single_qubit_gates):
        if i == 0:
            squ = HGate().to_matrix().dot(gate)
        elif i == len(single_qubit_gates) - 1:
            squ = gate.dot(UCGate._rz(np.pi / 2)).dot(HGate().to_matrix())
        else:
            squ = HGate().to_matrix().dot(gate.dot(UCGate._rz(np.pi / 2))).dot(HGate().to_matrix())
        circuit.unitary(squ, [q_target])
        binary_rep = np.binary_repr(i + 1)
        num_trailing_zeros = len(binary_rep) - len(binary_rep.rstrip('0'))
        q_contr_index = num_trailing_zeros
        if not i == len(single_qubit_gates) - 1:
            circuit.cx(q_controls[q_contr_index], q_target)
            circuit.global_phase -= 0.25 * np.pi
    if not self.up_to_diagonal:
        diagonal = Diagonal(diag)
        circuit.append(diagonal, q)
    return (circuit, diag)