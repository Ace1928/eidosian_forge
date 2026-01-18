import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
def test_run_non_terminal_measurement():
    sim = CountingSimulator()
    r = sim.run(cirq.Circuit(cirq.X(q0), cirq.measure(q0), cirq.X(q0)), repetitions=2)
    assert np.allclose(r.measurements['q(0)'], [[1], [1]])