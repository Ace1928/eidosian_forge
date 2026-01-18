from cmath import exp
from math import cos, sin, sqrt
import pytest
import numpy as np
from scipy.linalg import block_diag
from flaky import flaky
import pennylane as qml
@pytest.mark.parametrize('mat', [U, U2])
def test_qubit_unitary(self, device, init_state, mat, tol, skip_if, benchmark):
    """Test QubitUnitary gate."""
    n_wires = int(np.log2(len(mat)))
    dev = device(n_wires)
    if isinstance(dev, qml.Device):
        if 'QubitUnitary' not in dev.operations:
            pytest.skip('Skipped because device does not support QubitUnitary.')
        skip_if(dev, {'returns_probs': False})
    rnd_state = init_state(n_wires)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(rnd_state, wires=range(n_wires))
        qml.QubitUnitary(mat, wires=list(range(n_wires)))
        return qml.probs(wires=range(n_wires))
    res = benchmark(circuit)
    expected = np.abs(mat @ rnd_state) ** 2
    assert np.allclose(res, expected, atol=tol(dev.shots))