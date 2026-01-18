from typing import Sequence
import pytest
import duet
import numpy as np
import pandas as pd
import sympy
import cirq
def test_sampler_run_batch_default_params_and_repetitions():
    sampler = cirq.ZerosSampler()
    a = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.X(a), cirq.measure(a, key='m'))
    circuit2 = cirq.Circuit(cirq.Y(a), cirq.measure(a, key='m'))
    results = sampler.run_batch([circuit1, circuit2])
    assert len(results) == 2
    for result_list in results:
        assert len(result_list) == 1
        result = result_list[0]
        assert result.repetitions == 1
        assert result.params.param_dict == {}
        assert result.measurements == {'m': np.array([[0]], dtype='uint8')}