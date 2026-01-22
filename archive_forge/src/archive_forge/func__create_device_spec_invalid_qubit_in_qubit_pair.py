from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
def _create_device_spec_invalid_qubit_in_qubit_pair() -> v2.device_pb2.DeviceSpecification:
    """Creates a DeviceSpecification where qubit '0_1' is in a pair but not in valid_qubits."""
    q_proto_ids = [v2.qubit_to_proto_id(cirq.GridQubit(0, i)) for i in range(2)]
    spec = v2.device_pb2.DeviceSpecification()
    spec.valid_qubits.extend([q_proto_ids[0]])
    targets = spec.valid_targets.add()
    targets.name = 'test_targets'
    targets.target_ordering = v2.device_pb2.TargetSet.SYMMETRIC
    new_target = targets.targets.add()
    new_target.ids.extend([q_proto_ids[0], q_proto_ids[1]])
    return spec