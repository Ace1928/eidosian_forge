from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
def _create_device_spec_with_isolated_qubits():
    device_info, spec = _create_device_spec_with_horizontal_couplings()
    isolated_qubits = [cirq.GridQubit(GRID_HEIGHT, j) for j in range(2)]
    spec.valid_qubits.extend([v2.qubit_to_proto_id(q) for q in isolated_qubits])
    device_info.grid_qubits.extend(isolated_qubits)
    return (device_info, spec)