import numpy as np
import pytest
import matplotlib.pyplot as plt
import cirq
import cirq.experiments.qubit_characterizations as ceqc
from cirq import GridQubit
from cirq import circuits, ops, sim
from cirq.experiments import (
def test_two_qubit_state_tomography():
    simulator = sim.Simulator()
    q_0 = GridQubit(0, 0)
    q_1 = GridQubit(0, 1)
    circuit_00 = circuits.Circuit(ops.H(q_0), ops.CNOT(q_0, q_1))
    circuit_01 = circuits.Circuit(ops.X(q_1), ops.H(q_0), ops.CNOT(q_0, q_1))
    circuit_10 = circuits.Circuit(ops.X(q_0), ops.H(q_0), ops.CNOT(q_0, q_1))
    circuit_11 = circuits.Circuit(ops.X(q_0), ops.X(q_1), ops.H(q_0), ops.CNOT(q_0, q_1))
    circuit_hh = circuits.Circuit(ops.H(q_0), ops.H(q_1))
    circuit_xy = circuits.Circuit(ops.X(q_0) ** 0.5, ops.Y(q_1) ** 0.5)
    circuit_yx = circuits.Circuit(ops.Y(q_0) ** 0.5, ops.X(q_1) ** 0.5)
    act_rho_00 = two_qubit_state_tomography(simulator, q_0, q_1, circuit_00, 1000).data
    act_rho_01 = two_qubit_state_tomography(simulator, q_0, q_1, circuit_01, 1000).data
    act_rho_10 = two_qubit_state_tomography(simulator, q_0, q_1, circuit_10, 1000).data
    act_rho_11 = two_qubit_state_tomography(simulator, q_0, q_1, circuit_11, 1000).data
    act_rho_hh = two_qubit_state_tomography(simulator, q_0, q_1, circuit_hh, 1000).data
    act_rho_xy = two_qubit_state_tomography(simulator, q_0, q_1, circuit_xy, 1000).data
    act_rho_yx = two_qubit_state_tomography(simulator, q_0, q_1, circuit_yx, 1000).data
    tar_rho_00 = np.outer([1.0, 0, 0, 1.0], [1.0, 0, 0, 1.0]) * 0.5
    tar_rho_01 = np.outer([0, 1.0, 1.0, 0], [0, 1.0, 1.0, 0]) * 0.5
    tar_rho_10 = np.outer([1.0, 0, 0, -1.0], [1.0, 0, 0, -1.0]) * 0.5
    tar_rho_11 = np.outer([0, 1.0, -1.0, 0], [0, 1.0, -1.0, 0]) * 0.5
    tar_rho_hh = np.outer([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
    tar_rho_xy = np.outer([0.5, 0.5, -0.5j, -0.5j], [0.5, 0.5, 0.5j, 0.5j])
    tar_rho_yx = np.outer([0.5, -0.5j, 0.5, -0.5j], [0.5, 0.5j, 0.5, 0.5j])
    np.testing.assert_almost_equal(act_rho_00, tar_rho_00, decimal=1)
    np.testing.assert_almost_equal(act_rho_01, tar_rho_01, decimal=1)
    np.testing.assert_almost_equal(act_rho_10, tar_rho_10, decimal=1)
    np.testing.assert_almost_equal(act_rho_11, tar_rho_11, decimal=1)
    np.testing.assert_almost_equal(act_rho_hh, tar_rho_hh, decimal=1)
    np.testing.assert_almost_equal(act_rho_xy, tar_rho_xy, decimal=1)
    np.testing.assert_almost_equal(act_rho_yx, tar_rho_yx, decimal=1)