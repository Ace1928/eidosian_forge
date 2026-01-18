from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires('numpy')
def test_to_unitless___0D_array_with_object():
    from ..util._expr import Constant
    pi = np.array(Constant(np.pi))
    one_thousand = to_unitless(pi * u.metre, u.millimeter)
    assert get_physical_dimensionality(one_thousand) == {}
    assert abs(magnitude(one_thousand) - np.arctan(1) * 4000.0) < 1e-12