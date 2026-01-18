import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
def test_run_non_unitary_circuit_non_unitary_state():

    class DensityCountingSimulator(CountingSimulator):

        def _can_be_in_run_prefix(self, val):
            return not cirq.is_measurement(val)
    sim = DensityCountingSimulator()
    r = sim.run(cirq.Circuit(cirq.phase_damp(1).on(q0), cirq.measure(q0)), repetitions=2)
    assert np.allclose(r.measurements['q(0)'], [[1], [1]])