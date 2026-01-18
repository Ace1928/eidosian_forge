import pytest
import cirq
from cirq.testing.devices import ValidatingTestDevice
def test_validating_types_and_qubits():
    dev = ValidatingTestDevice(allowed_qubit_types=(cirq.GridQubit,), allowed_gates=(cirq.XPowGate,), qubits={cirq.GridQubit(0, 0)}, name='test')
    dev.validate_operation(cirq.X(cirq.GridQubit(0, 0)))
    with pytest.raises(ValueError, match='Unsupported qubit type'):
        dev.validate_operation(cirq.X(cirq.NamedQubit('a')))
    with pytest.raises(ValueError, match='Qubit not on device'):
        dev.validate_operation(cirq.X(cirq.GridQubit(1, 0)))
    with pytest.raises(ValueError, match='Unsupported gate type'):
        dev.validate_operation(cirq.Y(cirq.GridQubit(0, 0)))