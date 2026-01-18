import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import get_buffer_info
import pytest
from numpy.testing import assert_, assert_equal, assert_raises
def test_user_scalar_fails_buffer(self):
    r = rational(1)
    with assert_raises(TypeError):
        memoryview(r)
    with pytest.raises(BufferError, match='scalar buffer is readonly'):
        get_buffer_info(r, ['WRITABLE'])