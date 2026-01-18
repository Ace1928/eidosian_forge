from typing import Sequence
import pytest
import duet
import numpy as np
import pandas as pd
import sympy
import cirq
def test_sampler_sample_no_params():
    a, b = cirq.LineQubit.range(2)
    sampler = cirq.Simulator()
    circuit = cirq.Circuit(cirq.X(a), cirq.measure(a, b, key='out'))
    results = sampler.sample(circuit, repetitions=3)
    pd.testing.assert_frame_equal(results, pd.DataFrame(columns=['out'], index=[0, 1, 2], data=[[2], [2], [2]]))