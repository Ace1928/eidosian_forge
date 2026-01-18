from datetime import timedelta
from typing import List
import pytest
import cirq
from cirq_aqt import aqt_device, aqt_device_metadata
def test_validate_operation_supported_gate(device):

    class MyGate(cirq.Gate):

        def num_qubits(self):
            return 1
    device.validate_operation(cirq.GateOperation(cirq.Z, [cirq.LineQubit(0)]))
    assert MyGate().num_qubits() == 1
    with pytest.raises(ValueError):
        device.validate_operation(cirq.GateOperation(MyGate(), [cirq.LineQubit(0)]))
    with pytest.raises(ValueError):
        device.validate_operation(NotImplementedOperation())