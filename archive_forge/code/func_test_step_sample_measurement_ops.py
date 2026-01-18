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
def test_step_sample_measurement_ops():
    q0, q1, q2 = cirq.LineQubit.range(3)
    measurement_ops = [cirq.measure(q0, q1), cirq.measure(q2)]
    step_result = FakeStepResult(ones_qubits=[q1])
    measurements = step_result.sample_measurement_ops(measurement_ops)
    np.testing.assert_equal(measurements, {'q(0),q(1)': [[False, True]], 'q(2)': [[False]]})