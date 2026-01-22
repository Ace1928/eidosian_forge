from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
@dataclass
class _DeviceInfo:
    """Dataclass for device information relevant to GridDevice tests."""
    grid_qubits: List[cirq.GridQubit]
    qubit_pairs: List[Tuple[cirq.GridQubit, cirq.GridQubit]]
    expected_gateset: cirq.Gateset
    expected_gate_durations: Dict[cirq.GateFamily, cirq.Duration]
    expected_target_gatesets: Tuple[cirq.CompilationTargetGateset, ...]