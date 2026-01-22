from os.path import join, dirname
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
import pytest
from pytest import raises as assert_raises
from scipy.fftpack._realtransforms import (
class TestIDSTIFloat(_TestIDSTBase):

    def setup_method(self):
        self.rdt = np.float32
        self.dec = 4
        self.type = 1