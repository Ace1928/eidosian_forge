from cmath import exp
from math import cos, sin, sqrt
import pytest
import numpy as np
from scipy.linalg import block_diag
from flaky import flaky
import pennylane as qml
@pytest.mark.parametrize('op,mat', single_qubit)
def test_single_qubit_no_parameters(self, device, init_state, op, mat, tol, skip_if, benchmark):
    """Test PauliX application."""
    n_wires = 1
    dev = device(n_wires)
    if isinstance(dev, qml.Device):
        skip_if(dev, {'returns_probs': False})
    rnd_state = init_state(n_wires)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(rnd_state, wires=range(n_wires))
        op(wires=range(n_wires))
        return qml.probs(wires=range(n_wires))
    res = benchmark(circuit)
    expected = np.abs(mat @ rnd_state) ** 2
    assert np.allclose(res, expected, atol=tol(dev.shots))