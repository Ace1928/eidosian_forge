import numpy as np
import pytest
import matplotlib.pyplot as plt
import cirq
import cirq.experiments.qubit_characterizations as ceqc
from cirq import GridQubit
from cirq import circuits, ops, sim
from cirq.experiments import (
def test_single_qubit_state_tomography():
    simulator = sim.Simulator()
    qubit = GridQubit(0, 0)
    circuit_1 = circuits.Circuit(ops.X(qubit) ** 0.5)
    circuit_2 = circuits.Circuit(ops.Y(qubit) ** 0.5)
    circuit_3 = circuits.Circuit(ops.H(qubit), ops.Y(qubit))
    act_rho_1 = single_qubit_state_tomography(simulator, qubit, circuit_1, 1000).data
    act_rho_2 = single_qubit_state_tomography(simulator, qubit, circuit_2, 1000).data
    act_rho_3 = single_qubit_state_tomography(simulator, qubit, circuit_3, 1000).data
    tar_rho_1 = np.array([[0.5, 0.5j], [-0.5j, 0.5]])
    tar_rho_2 = np.array([[0.5, 0.5], [0.5, 0.5]])
    tar_rho_3 = np.array([[0.5, -0.5], [-0.5, 0.5]])
    np.testing.assert_almost_equal(act_rho_1, tar_rho_1, decimal=1)
    np.testing.assert_almost_equal(act_rho_2, tar_rho_2, decimal=1)
    np.testing.assert_almost_equal(act_rho_3, tar_rho_3, decimal=1)