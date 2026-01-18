from typing import Sequence
import pytest
import duet
import numpy as np
import pandas as pd
import sympy
import cirq
def test_sampler_run_batch_bad_input_lengths():
    sampler = cirq.ZerosSampler()
    a = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.X(a) ** sympy.Symbol('t'), cirq.measure(a, key='m'))
    circuit2 = cirq.Circuit(cirq.Y(a) ** sympy.Symbol('t'), cirq.measure(a, key='m'))
    params1 = cirq.Points('t', [0.3, 0.7])
    params2 = cirq.Points('t', [0.4, 0.6])
    with pytest.raises(ValueError, match='2 and 1'):
        _ = sampler.run_batch([circuit1, circuit2], params_list=[params1])
    with pytest.raises(ValueError, match='2 and 3'):
        _ = sampler.run_batch([circuit1, circuit2], params_list=[params1, params2], repetitions=[1, 2, 3])