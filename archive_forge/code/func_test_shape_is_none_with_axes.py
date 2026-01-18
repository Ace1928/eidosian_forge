from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
@pytest.mark.parametrize('fforward,finverse', [(dctn, idctn), (dstn, idstn)])
@pytest.mark.parametrize('axes', [1, (1,), [1], 0, (0,), [0]])
def test_shape_is_none_with_axes(self, fforward, finverse, axes):
    tmp = fforward(self.data, s=None, axes=axes, norm='ortho')
    tmp = finverse(tmp, s=None, axes=axes, norm='ortho')
    assert_array_almost_equal(self.data, tmp, decimal=self.dec)