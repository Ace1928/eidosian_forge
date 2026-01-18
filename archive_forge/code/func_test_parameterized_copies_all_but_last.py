import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
def test_parameterized_copies_all_but_last():
    sim = CountingSimulator()
    n = 4
    rs = sim.simulate_sweep(cirq.Circuit(cirq.X(q0) ** 'a'), [{'a': i} for i in range(n)])
    for i in range(n):
        r = rs[i]
        assert r._final_simulator_state.gate_count == 1
        assert r._final_simulator_state.measurement_count == 0
        assert r._final_simulator_state.copy_count == 0 if i == n - 1 else 1