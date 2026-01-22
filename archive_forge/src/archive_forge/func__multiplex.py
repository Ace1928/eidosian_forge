from typing import Union, Optional
import math
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates.x import CXGate, XGate
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.circuit.library.standard_gates.s import SGate, SdgGate
from qiskit.circuit.library.standard_gates.ry import RYGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info.states.statevector import Statevector  # pylint: disable=cyclic-import
def _multiplex(self, target_gate, list_of_angles, last_cnot=True):
    """
        Return a recursive implementation of a multiplexor circuit,
        where each instruction itself has a decomposition based on
        smaller multiplexors.

        The LSB is the multiplexor "data" and the other bits are multiplexor "select".

        Args:
            target_gate (Gate): Ry or Rz gate to apply to target qubit, multiplexed
                over all other "select" qubits
            list_of_angles (list[float]): list of rotation angles to apply Ry and Rz
            last_cnot (bool): add the last cnot if last_cnot = True

        Returns:
            DAGCircuit: the circuit implementing the multiplexor's action
        """
    list_len = len(list_of_angles)
    local_num_qubits = int(math.log2(list_len)) + 1
    q = QuantumRegister(local_num_qubits)
    circuit = QuantumCircuit(q, name='multiplex' + str(local_num_qubits))
    lsb = q[0]
    msb = q[local_num_qubits - 1]
    if local_num_qubits == 1:
        circuit.append(target_gate(list_of_angles[0]), [q[0]])
        return circuit
    angle_weight = np.kron([[0.5, 0.5], [0.5, -0.5]], np.identity(2 ** (local_num_qubits - 2)))
    list_of_angles = angle_weight.dot(np.array(list_of_angles)).tolist()
    multiplex_1 = self._multiplex(target_gate, list_of_angles[0:list_len // 2], False)
    circuit.append(multiplex_1.to_instruction(), q[0:-1])
    circuit.append(CXGate(), [msb, lsb])
    multiplex_2 = self._multiplex(target_gate, list_of_angles[list_len // 2:], False)
    if list_len > 1:
        circuit.append(multiplex_2.to_instruction().reverse_ops(), q[0:-1])
    else:
        circuit.append(multiplex_2.to_instruction(), q[0:-1])
    if last_cnot:
        circuit.append(CXGate(), [msb, lsb])
    return circuit