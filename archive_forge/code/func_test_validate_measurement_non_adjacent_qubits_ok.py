from datetime import timedelta
from typing import List
import pytest
import cirq
from cirq_aqt import aqt_device, aqt_device_metadata
def test_validate_measurement_non_adjacent_qubits_ok(device):
    device.validate_operation(cirq.GateOperation(cirq.MeasurementGate(2, 'key'), (cirq.LineQubit(0), cirq.LineQubit(1))))