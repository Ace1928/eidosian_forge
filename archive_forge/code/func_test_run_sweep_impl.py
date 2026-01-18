from typing import Sequence
import pytest
import duet
import numpy as np
import pandas as pd
import sympy
import cirq
def test_run_sweep_impl():
    """Test run_sweep implemented in terms of run_sweep_async."""

    class AsyncSampler(cirq.Sampler):

        async def run_sweep_async(self, program, params, repetitions: int=1):
            await duet.sleep(0.001)
            return cirq.Simulator().run_sweep(program, params, repetitions)
    results = AsyncSampler().run_sweep(cirq.Circuit(cirq.measure(cirq.GridQubit(0, 0), key='m')), cirq.Linspace('foo', 0, 1, 10), repetitions=10)
    assert len(results) == 10
    for result in results:
        np.testing.assert_equal(result.records['m'], np.zeros((10, 1, 1)))