import pytest
import cirq
class NotImplBoth:

    def _num_qubits_(self):
        return NotImplemented

    def _qid_shape_(self):
        return NotImplemented