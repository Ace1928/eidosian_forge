import pytest
import cirq
class ExpectsArgsQubits:

    def _qasm_(self, args, qubits):
        return 'text'