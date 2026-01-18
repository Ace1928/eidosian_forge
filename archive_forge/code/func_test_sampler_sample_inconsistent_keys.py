from typing import Sequence
import pytest
import duet
import numpy as np
import pandas as pd
import sympy
import cirq
def test_sampler_sample_inconsistent_keys():
    q = cirq.LineQubit(0)
    sampler = cirq.Simulator()
    circuit = cirq.Circuit(cirq.measure(q, key='out'))
    with pytest.raises(ValueError, match='Inconsistent sweep parameters'):
        _ = sampler.sample(circuit, params=[{'a': 1}, {'a': 1, 'b': 2}])