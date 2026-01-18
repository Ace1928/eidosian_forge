import numpy as np
import pytest
import matplotlib.pyplot as plt
import cirq
import cirq.experiments.qubit_characterizations as ceqc
from cirq import GridQubit
from cirq import circuits, ops, sim
from cirq.experiments import (
def test_single_qubit_randomized_benchmarking():
    simulator = sim.Simulator()
    qubit = GridQubit(0, 0)
    num_cfds = range(5, 20, 5)
    results = single_qubit_randomized_benchmarking(simulator, qubit, num_clifford_range=num_cfds, repetitions=100)
    g_pops = np.asarray(results.data)[:, 1]
    assert np.isclose(np.mean(g_pops), 1.0)