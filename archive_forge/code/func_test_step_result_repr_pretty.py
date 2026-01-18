import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_step_result_repr_pretty():
    q0 = cirq.LineQubit(0)
    sim = ccq.mps_simulator.MPSSimulator()
    step_result = next(sim.simulate_moment_steps(cirq.Circuit(cirq.measure(q0))))
    cirq.testing.assert_repr_pretty_contains(step_result, 'TensorNetwork')
    cirq.testing.assert_repr_pretty(step_result, 'cirq.MPSSimulatorStepResult(...)', cycle=True)