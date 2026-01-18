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
@pytest.mark.parametrize('qubit', [cirq.GridQubit(0, 0), cirq.GridQubit(1, 1), cirq.LineQubit(30), cirq.NamedQubit('33'), AspenQubit(3, 6), OctagonalQubit(6)])
def test_rigetti_qcs_aspen_device_valid_qubit(qubit: cirq.Qid, qcs_aspen8_isa: InstructionSetArchitecture):
    """test RigettiQCSAspenDevice throws no error on valid qubits or operations on those qubits"""
    device = RigettiQCSAspenDevice(isa=qcs_aspen8_isa)
    device.validate_qubit(qubit)
    device.validate_operation(cirq.I(qubit))