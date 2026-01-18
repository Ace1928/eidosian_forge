import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
def test_output_fill_error(self):
    rg = self.rg
    size = (31, 7, 97)
    existing = np.empty(size)
    with pytest.raises(TypeError):
        rg.standard_normal(out=existing, dtype=np.float32)
    with pytest.raises(ValueError):
        rg.standard_normal(out=existing[::3])
    existing = np.empty(size, dtype=np.float32)
    with pytest.raises(TypeError):
        rg.standard_normal(out=existing, dtype=np.float64)
    existing = np.zeros(size, dtype=np.float32)
    with pytest.raises(TypeError):
        rg.standard_gamma(1.0, out=existing, dtype=np.float64)
    with pytest.raises(ValueError):
        rg.standard_gamma(1.0, out=existing[::3], dtype=np.float32)
    existing = np.zeros(size, dtype=np.float64)
    with pytest.raises(TypeError):
        rg.standard_gamma(1.0, out=existing, dtype=np.float32)
    with pytest.raises(ValueError):
        rg.standard_gamma(1.0, out=existing[::3])