import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
def test_sfloat_promotion(self):
    assert np.result_type(SF(2.0), SF(3.0)) == SF(3.0)
    assert np.result_type(SF(3.0), SF(2.0)) == SF(3.0)
    assert np.result_type(SF(3.0), np.float64) == SF(3.0)
    assert np.result_type(np.float64, SF(0.5)) == SF(1.0)
    with pytest.raises(TypeError):
        np.result_type(SF(1.0), np.int64)