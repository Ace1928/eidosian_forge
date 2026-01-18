import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
@pytest.mark.parametrize('aligned', [True, False])
@pytest.mark.parametrize('scaling', [1.0, -1.0, 2.0])
def test_sfloat_getitem(self, aligned, scaling):
    a = self._get_array(1.0, aligned)
    assert a.tolist() == [1.0, 2.0, 3.0]