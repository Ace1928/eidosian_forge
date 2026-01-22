import pytest
import cirq
class NotImplNum:

    def _num_qubits_(self):
        return NotImplemented