import pytest
import cirq
def test_two_qubit_gate_is_abstract_can_implement():

    class Included(cirq.testing.TwoQubitGate):

        def matrix(self):
            pass
    assert isinstance(Included(), cirq.testing.TwoQubitGate)