from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library)
def test_polyfit_polyval():
    p1 = pnp.polyfit([0, 1, 2], [0, 1, 4], 2)
    assert allclose(p1, [1, 0, 0], atol=1e-14)
    assert allclose(pnp.polyval(p1, 3), 9)
    assert allclose(pnp.polyval(p1, [4, 5]), [16, 25])
    p2 = pnp.polyfit([0, 1, 2] * u.s, [0, 1, 4] * u.m, 2)
    for _p, _r, _a in zip(p2, [1 * u.m / u.s ** 2, 0 * u.m / u.s, 0 * u.m], [0 * u.m / u.s ** 2, 1e-15 * u.m / u.s, 1e-15 * u.m]):
        assert allclose(_p, _r, atol=_a)
    assert allclose(pnp.polyval(p2, 3 * u.s), 9 * u.m)
    assert allclose(pnp.polyval(p2, [4, 5] * u.s), [16, 25] * u.m)