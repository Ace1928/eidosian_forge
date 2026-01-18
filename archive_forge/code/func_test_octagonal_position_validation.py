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
def test_octagonal_position_validation():
    """test OctagonalQubit validates octagon position when initialized"""
    with pytest.raises(ValueError):
        _ = OctagonalQubit(8)