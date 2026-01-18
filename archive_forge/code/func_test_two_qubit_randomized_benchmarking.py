import numpy as np
import pytest
import matplotlib.pyplot as plt
import cirq
import cirq.experiments.qubit_characterizations as ceqc
from cirq import GridQubit
from cirq import circuits, ops, sim
from cirq.experiments import (
def test_two_qubit_randomized_benchmarking():
    simulator = sim.Simulator()
    q_0 = GridQubit(0, 0)
    q_1 = GridQubit(0, 1)
    num_cfds = [5, 10]
    results = two_qubit_randomized_benchmarking(simulator, q_0, q_1, num_clifford_range=num_cfds, num_circuits=10, repetitions=100)
    g_pops = np.asarray(results.data)[:, 1]
    assert np.isclose(np.mean(g_pops), 1.0)