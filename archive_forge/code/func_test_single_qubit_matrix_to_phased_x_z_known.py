import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
def test_single_qubit_matrix_to_phased_x_z_known():
    actual = cirq.single_qubit_matrix_to_phased_x_z(np.array([[0, 1], [1, 0]]), atol=0.01)
    assert cirq.approx_eq(actual, [cirq.PhasedXPowGate(phase_exponent=1.0)], atol=1e-09)
    actual = cirq.single_qubit_matrix_to_phased_x_z(np.array([[0, -1j], [1j, 0]]), atol=0.01)
    assert cirq.approx_eq(actual, [cirq.PhasedXPowGate(phase_exponent=0.5, exponent=-1)], atol=1e-09)
    actual = cirq.single_qubit_matrix_to_phased_x_z(np.array([[1, 0], [0, -1]]), atol=0.01)
    assert cirq.approx_eq(actual, [cirq.Z], atol=1e-09)
    actual = cirq.single_qubit_matrix_to_phased_x_z(np.array([[1, 0], [0, 1j]]), atol=0.01)
    assert cirq.approx_eq(actual, [cirq.Z ** 0.5], atol=1e-09)
    actual = cirq.single_qubit_matrix_to_phased_x_z(np.array([[1, 0], [0, -1j]]), atol=0.01)
    assert cirq.approx_eq(actual, [cirq.Z ** (-0.5)], atol=1e-09)
    actual = cirq.single_qubit_matrix_to_phased_x_z(np.array([[1, 1], [1, -1]]) * np.sqrt(0.5), atol=0.001)
    assert cirq.approx_eq(actual, [cirq.PhasedXPowGate(phase_exponent=-0.5, exponent=0.5), cirq.Z ** (-1)], atol=1e-09)