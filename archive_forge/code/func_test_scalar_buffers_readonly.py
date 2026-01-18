import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import get_buffer_info
import pytest
from numpy.testing import assert_, assert_equal, assert_raises
@pytest.mark.parametrize('scalar', scalars_only, ids=codes_only)
def test_scalar_buffers_readonly(self, scalar):
    x = scalar()
    with pytest.raises(BufferError, match='scalar buffer is readonly'):
        get_buffer_info(x, ['WRITABLE'])