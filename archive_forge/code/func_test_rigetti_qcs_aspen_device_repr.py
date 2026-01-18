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
def test_rigetti_qcs_aspen_device_repr(qcs_aspen8_isa: InstructionSetArchitecture):
    """test RigettiQCSAspenDevice.__repr__"""
    device = RigettiQCSAspenDevice(isa=qcs_aspen8_isa)
    assert f'cirq_rigetti.RigettiQCSAspenDevice(isa={qcs_aspen8_isa!r})' == repr(device)