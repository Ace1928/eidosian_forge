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
def _dec_ucg_help(self):
    """
        This method finds the single qubit gate arising in the decomposition of UCGates given in
        https://arxiv.org/pdf/quant-ph/0410066.pdf.
        """
    single_qubit_gates = [gate.astype(complex) for gate in self.params]
    diag = np.ones(2 ** self.num_qubits, dtype=complex)
    num_contr = self.num_qubits - 1
    for dec_step in range(num_contr):
        num_ucgs = 2 ** dec_step
        for ucg_index in range(num_ucgs):
            len_ucg = 2 ** (num_contr - dec_step)
            for i in range(int(len_ucg / 2)):
                shift = ucg_index * len_ucg
                a = single_qubit_gates[shift + i]
                b = single_qubit_gates[shift + len_ucg // 2 + i]
                v, u, r = self._demultiplex_single_uc(a, b)
                single_qubit_gates[shift + i] = v
                single_qubit_gates[shift + len_ucg // 2 + i] = u
                if ucg_index < num_ucgs - 1:
                    k = shift + len_ucg + i
                    single_qubit_gates[k] = single_qubit_gates[k].dot(UCGate._ct(r)) * UCGate._rz(np.pi / 2).item((0, 0))
                    k = k + len_ucg // 2
                    single_qubit_gates[k] = single_qubit_gates[k].dot(r) * UCGate._rz(np.pi / 2).item((1, 1))
                else:
                    for ucg_index_2 in range(num_ucgs):
                        shift_2 = ucg_index_2 * len_ucg
                        k = 2 * (i + shift_2)
                        diag[k] = diag[k] * UCGate._ct(r).item((0, 0)) * UCGate._rz(np.pi / 2).item((0, 0))
                        diag[k + 1] = diag[k + 1] * UCGate._ct(r).item((1, 1)) * UCGate._rz(np.pi / 2).item((0, 0))
                        k = len_ucg + k
                        diag[k] *= r.item((0, 0)) * UCGate._rz(np.pi / 2).item((1, 1))
                        diag[k + 1] *= r.item((1, 1)) * UCGate._rz(np.pi / 2).item((1, 1))
    return (single_qubit_gates, diag)