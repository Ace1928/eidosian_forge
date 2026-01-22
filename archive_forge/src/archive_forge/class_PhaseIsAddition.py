import pytest
import cirq
class PhaseIsAddition:

    def __init__(self, num_qubits):
        self.phase = [0] * num_qubits
        self.num_qubits = num_qubits

    def _phase_by_(self, phase_turns, qubit_on):
        if qubit_on >= self.num_qubits:
            return self
        self.phase[qubit_on] += phase_turns
        return self