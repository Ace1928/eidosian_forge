import abc
from typing import Generic, Dict, Any, List, Sequence, Union
from unittest import mock
import duet
import numpy as np
import pytest
import cirq
from cirq import study
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulator import (
def test_step_sample_measurement_ops_confusion_map():
    q0, q1, q2 = cirq.LineQubit.range(3)
    cmap_01 = {(0, 1): np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])}
    cmap_2 = {(0,): np.array([[0, 1], [1, 0]])}
    measurement_ops = [cirq.measure(q0, q1, confusion_map=cmap_01), cirq.measure(q2, confusion_map=cmap_2)]
    step_result = FakeStepResult(ones_qubits=[q2])
    measurements = step_result.sample_measurement_ops(measurement_ops)
    np.testing.assert_equal(measurements, {'q(0),q(1)': [[False, True]], 'q(2)': [[False]]})