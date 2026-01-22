from os.path import join, dirname
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
import pytest
from pytest import raises as assert_raises
from scipy.fftpack._realtransforms import (
class _TestDCTIIIBase(_TestDCTBase):

    def test_definition_ortho(self):
        dt = np.result_type(np.float32, self.rdt)
        for xr in X:
            x = np.array(xr, dtype=self.rdt)
            y = dct(x, norm='ortho', type=2)
            xi = dct(y, norm='ortho', type=3)
            assert_equal(xi.dtype, dt)
            assert_array_almost_equal(xi, x, decimal=self.dec)