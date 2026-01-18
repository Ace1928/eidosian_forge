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
def test_rigetti_qcs_aspen_device_qubits(qcs_aspen8_isa: InstructionSetArchitecture):
    """test RigettiQCSAspenDevice returns accurate set of qubits"""
    device = RigettiQCSAspenDevice(isa=qcs_aspen8_isa)
    expected_qubits = set()
    for i in range(4):
        for j in range(8):
            expected_qubits.add(AspenQubit(octagon=i, octagon_position=j))
    assert expected_qubits == set(device.qubits())