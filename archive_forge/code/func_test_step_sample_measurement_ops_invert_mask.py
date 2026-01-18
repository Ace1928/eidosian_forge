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
def test_step_sample_measurement_ops_invert_mask():
    q0, q1, q2 = cirq.LineQubit.range(3)
    measurement_ops = [cirq.measure(q0, q1, invert_mask=(True,)), cirq.measure(q2, invert_mask=(False,))]
    step_result = FakeStepResult(ones_qubits=[q1])
    measurements = step_result.sample_measurement_ops(measurement_ops)
    np.testing.assert_equal(measurements, {'q(0),q(1)': [[True, True]], 'q(2)': [[False]]})