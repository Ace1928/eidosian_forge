import pytest
import cirq
def _phase_by_(self, phase_turns, qubit_on):
    if qubit_on >= self.num_qubits:
        return self
    self.phase[qubit_on] += phase_turns
    return self