from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library)
def test_patched_numpy():
    assert allclose(pnp.exp(3 * u.joule / (2 * u.cal)), 1.43119335, rtol=1e-05)
    for arg in ([1, 2], [[1], [2]], [1], 2):
        assert np.all(pnp.exp(arg) == np.exp(arg))