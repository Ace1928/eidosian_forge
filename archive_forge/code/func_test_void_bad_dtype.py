import pytest
import numpy as np
from numpy.testing import (
def test_void_bad_dtype():
    with pytest.raises(TypeError, match='void: descr must be a `void.*int64'):
        np.void(4, dtype='i8')
    with pytest.raises(TypeError, match='void: descr must be a `void.*\\(4,\\)'):
        np.void(4, dtype='4i')