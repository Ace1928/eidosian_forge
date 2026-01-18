import pytest
import cirq
def test_phase_by():

    class NoMethod:
        pass

    class ReturnsNotImplemented:

        def _phase_by_(self, phase_turns, qubit_on):
            return NotImplemented

    class PhaseIsAddition:

        def __init__(self, num_qubits):
            self.phase = [0] * num_qubits
            self.num_qubits = num_qubits

        def _phase_by_(self, phase_turns, qubit_on):
            if qubit_on >= self.num_qubits:
                return self
            self.phase[qubit_on] += phase_turns
            return self
    n = NoMethod()
    rin = ReturnsNotImplemented()
    with pytest.raises(TypeError, match='no _phase_by_ method'):
        _ = cirq.phase_by(n, 1, 0)
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = cirq.phase_by(rin, 1, 0)
    assert cirq.phase_by(n, 1, 0, default=None) is None
    assert cirq.phase_by(rin, 1, 0, default=None) is None
    test = PhaseIsAddition(3)
    assert test.phase == [0, 0, 0]
    test = cirq.phase_by(test, 0.25, 0)
    assert test.phase == [0.25, 0, 0]
    test = cirq.phase_by(test, 0.25, 0)
    assert test.phase == [0.5, 0, 0]
    test = cirq.phase_by(test, 0.4, 1)
    assert test.phase == [0.5, 0.4, 0]
    test = cirq.phase_by(test, 0.4, 4)
    assert test.phase == [0.5, 0.4, 0]