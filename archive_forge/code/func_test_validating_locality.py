import pytest
import cirq
from cirq.testing.devices import ValidatingTestDevice
def test_validating_locality():
    dev = ValidatingTestDevice(allowed_qubit_types=(cirq.GridQubit,), allowed_gates=(cirq.CZPowGate, cirq.MeasurementGate), qubits=set(cirq.GridQubit.rect(3, 3)), name='test', validate_locality=True)
    dev.validate_operation(cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)))
    dev.validate_operation(cirq.measure(cirq.GridQubit(0, 0), cirq.GridQubit(0, 2)))
    with pytest.raises(ValueError, match='Non-local interaction'):
        dev.validate_operation(cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(0, 2)))
    with pytest.raises(ValueError, match='GridQubit must be an allowed qubit type'):
        ValidatingTestDevice(allowed_qubit_types=(cirq.NamedQubit,), allowed_gates=(cirq.CZPowGate, cirq.MeasurementGate), qubits=set(cirq.GridQubit.rect(3, 3)), name='test', validate_locality=True)