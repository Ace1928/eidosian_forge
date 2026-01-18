import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_phase_by_xy():
    assert cirq.phase_by(cirq.X, 0.25, 0) == cirq.Y
    assert cirq.phase_by(cirq.X ** 0.5, 0.25, 0) == cirq.Y ** 0.5
    assert cirq.phase_by(cirq.X ** (-0.5), 0.25, 0) == cirq.Y ** (-0.5)