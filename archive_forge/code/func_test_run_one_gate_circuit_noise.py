import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
def test_run_one_gate_circuit_noise():
    sim = CountingSimulator(noise=cirq.X)
    r = sim.run(cirq.Circuit(cirq.X(q0), cirq.measure(q0)), repetitions=2)
    assert np.allclose(r.measurements['q(0)'], [[2], [2]])