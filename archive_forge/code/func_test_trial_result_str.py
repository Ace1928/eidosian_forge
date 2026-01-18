import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_trial_result_str():
    q0 = cirq.LineQubit(0)
    final_simulator_state = ccq.mps_simulator.MPSState(qubits=(q0,), prng=value.parse_random_state(0), simulation_options=ccq.mps_simulator.MPSOptions())
    result = ccq.mps_simulator.MPSTrialResult(params=cirq.ParamResolver({}), measurements={'m': np.array([[1]])}, final_simulator_state=final_simulator_state)
    assert 'output state: TensorNetwork' in str(result)