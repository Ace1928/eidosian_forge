import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
@pytest.mark.parametrize('aligned', [True, False])
def test_sfloat_casts(self, aligned):
    a = self._get_array(1.0, aligned)
    assert np.can_cast(a, SF(-1.0), casting='equiv')
    assert not np.can_cast(a, SF(-1.0), casting='no')
    na = a.astype(SF(-1.0))
    assert_array_equal(-1 * na.view(np.float64), a.view(np.float64))
    assert np.can_cast(a, SF(2.0), casting='same_kind')
    assert not np.can_cast(a, SF(2.0), casting='safe')
    a2 = a.astype(SF(2.0))
    assert_array_equal(2 * a2.view(np.float64), a.view(np.float64))