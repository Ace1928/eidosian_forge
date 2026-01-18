import os
from unittest.mock import patch, PropertyMock
from math import sqrt
import pathlib
import json
import pytest
import cirq
from cirq_rigetti import (
from qcs_api_client.models import InstructionSetArchitecture, Node
import numpy as np
@pytest.mark.parametrize('operation', [cirq.CNOT(OctagonalQubit(0), OctagonalQubit(2)), cirq.CNOT(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)), cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(2)), cirq.CNOT(cirq.NamedQubit('0'), cirq.NamedQubit('2')), cirq.CNOT(AspenQubit(0, 1), AspenQubit(1, 1))])
def test_rigetti_qcs_aspen_device_invalid_operation(operation: cirq.Operation, qcs_aspen8_isa: InstructionSetArchitecture):
    """test RigettiQCSAspenDevice throws error when validating 2Q operations on
    non-adjacent qubits
    """
    device = RigettiQCSAspenDevice(isa=qcs_aspen8_isa)
    with pytest.raises(UnsupportedRigettiQCSOperation):
        device.validate_operation(operation)