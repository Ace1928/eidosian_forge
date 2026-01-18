import cirq
import cirq_ionq as ionq
import pytest
from cirq_ionq.ionq_gateset_test import VALID_GATES
@pytest.mark.parametrize('gate', INVALID_GATES)
def test_validate_operation_invalid(gate):
    qubits = cirq.LineQubit.range(gate.num_qubits())
    device = ionq.IonQAPIDevice(qubits=qubits)
    operation = gate(*qubits)
    with pytest.raises(ValueError, match='unsupported gate'):
        device.validate_operation(operation)