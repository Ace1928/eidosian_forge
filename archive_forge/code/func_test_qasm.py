import pytest
import cirq
def test_qasm():
    assert cirq.qasm(NoMethod(), default=None) is None
    assert cirq.qasm(NoMethod(), default=5) == 5
    assert cirq.qasm(ReturnsText()) == 'text'
    with pytest.raises(TypeError, match='no _qasm_ method'):
        _ = cirq.qasm(NoMethod())
    with pytest.raises(TypeError, match='returned NotImplemented or None'):
        _ = cirq.qasm(ReturnsNotImplemented())
    assert cirq.qasm(ExpectsArgs(), args=cirq.QasmArgs()) == 'text'
    assert cirq.qasm(ExpectsArgsQubits(), args=cirq.QasmArgs(), qubits=()) == 'text'