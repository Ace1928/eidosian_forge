from cmath import exp
from math import cos, sin, sqrt
import pytest
import numpy as np
from scipy.linalg import block_diag
from flaky import flaky
import pennylane as qml
def test_state_prep(self, device, init_state, tol, skip_if):
    """Test StatePrep initialisation."""
    n_wires = 1
    dev = device(n_wires)
    if isinstance(dev, qml.Device):
        skip_if(dev, {'returns_probs': False})
    rnd_state = init_state(n_wires)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(rnd_state, wires=range(n_wires))
        return qml.probs(range(n_wires))
    res = circuit()
    expected = np.abs(rnd_state) ** 2
    assert np.allclose(res, expected, atol=tol(dev.shots))