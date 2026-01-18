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
def test_octagonal_qubit_index():
    """test that OctagonalQubit properly calculates index and uses it for comparison"""
    qubit0 = OctagonalQubit(0)
    assert qubit0.index == 0
    assert OctagonalQubit(1) > qubit0