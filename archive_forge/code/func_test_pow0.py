from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library)
def test_pow0():
    a = [1, 2] * u.metre
    b = a ** 0
    assert allclose(b, [1, 1])
    c = a ** 2
    assert allclose(c, [1, 4] * u.m ** 2)