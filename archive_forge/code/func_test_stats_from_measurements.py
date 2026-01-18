import dataclasses
import datetime
import time
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import _MeasurementSpec
def test_stats_from_measurements():
    bitstrings = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    qubit_to_index = {a: 0, b: 1}
    obs = cirq.Z(a) * cirq.Z(b) * 10
    mean, err = _stats_from_measurements(bitstrings, qubit_to_index, obs, atol=1e-08)
    assert mean == 0
    assert err == 10 ** 2 / (4 - 1)