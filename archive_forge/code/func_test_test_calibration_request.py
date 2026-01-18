from typing import Iterable, Optional, Tuple
import collections
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq_google
from cirq_google.calibration.engine_simulator import (
from cirq_google.calibration import (
import cirq
def test_test_calibration_request():
    a, b = cirq.LineQubit.range(2)
    request = ExamplePhasedFSimCalibrationRequest(gate=cirq.FSimGate(np.pi / 4, 0.5), pairs=((a, b),), options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION)
    assert request.to_calibration_layer() is NotImplemented
    result = mock.MagicMock(spec=cirq_google.CalibrationResult)
    assert request.parse_result(result) is NotImplemented