import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
@pytest.mark.parametrize('observable', all_obs)
def test_supported_observables_can_be_implemented(self, device_kwargs, observable):
    """Test that the device can implement all its supported observables."""
    device_kwargs['wires'] = 3
    dev = qml.device(**device_kwargs)
    if dev.shots and observable == 'SparseHamiltonian':
        pytest.skip('SparseHamiltonian only supported in analytic mode')
    if isinstance(dev, qml.Device):
        assert hasattr(dev, 'observables')
        if observable not in dev.observables:
            pytest.skip('observable not supported')
    kwargs = {'diff_method': 'parameter-shift'} if observable == 'SparseHamiltonian' else {}

    @qml.qnode(dev, **kwargs)
    def circuit(obs_circ):
        qml.PauliX(0)
        return qml.expval(obs_circ)
    if observable == 'Projector':
        for o in obs[observable]:
            assert isinstance(circuit(o), (float, np.ndarray))
    else:
        assert isinstance(circuit(obs[observable]), (float, np.ndarray))