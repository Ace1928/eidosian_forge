import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
@pytest.mark.parametrize('aligned', [True, False])
def test_sfloat_cast_internal_errors(self, aligned):
    a = self._get_array(2e+300, aligned)
    with pytest.raises(TypeError, match='error raised inside the core-loop: non-finite factor!'):
        a.astype(SF(2e-300))