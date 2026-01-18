import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
@pytest.mark.parametrize('axis', [None, 0])
def test_string_dtype_does_not_inspect(self, axis):
    with pytest.raises(TypeError):
        np.concatenate(([None], [1]), dtype='S', axis=axis)
    with pytest.raises(TypeError):
        np.concatenate(([None], [1]), dtype='U', axis=axis)