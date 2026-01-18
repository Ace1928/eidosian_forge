import pytest
import numpy as np
from cirq.sim import simulation_utils
from cirq import testing
@pytest.mark.parametrize('n,m', [(n, m) for n in range(1, 4) for m in range(1, n + 1)])
def test_state_probabilities_by_indices(n: int, m: int):
    np.random.seed(0)
    state = testing.random_superposition(1 << n)
    d = (state.conj() * state).real
    desired_axes = list(np.random.choice(n, m, replace=False))
    not_wanted = [i for i in range(n) if i not in desired_axes]
    got = simulation_utils.state_probabilities_by_indices(d, desired_axes, (2,) * n)
    want = np.transpose(d.reshape((2,) * n), desired_axes + not_wanted)
    want = np.sum(want.reshape((1 << len(desired_axes), -1)), axis=-1)
    np.testing.assert_allclose(want, got)