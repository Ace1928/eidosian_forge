import pytest
import cirq
def test_three_qubit_gate_validate():

    class Example(cirq.testing.ThreeQubitGate):

        def matrix(self):
            pass
    g = Example()
    a, b, c, d = cirq.LineQubit.range(4)
    assert g.num_qubits() == 3
    g.validate_args([a, b, c])
    with pytest.raises(ValueError):
        g.validate_args([])
    with pytest.raises(ValueError):
        g.validate_args([a])
    with pytest.raises(ValueError):
        g.validate_args([a, b])
    with pytest.raises(ValueError):
        g.validate_args([a, b, c, d])