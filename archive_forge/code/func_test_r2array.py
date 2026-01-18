import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
def test_r2array(self):
    """ Test to make sure equivalent Travis O's r2array function
        """
    assert_(atleast_2d(3).shape == (1, 1))
    assert_(atleast_2d([3j, 1]).shape == (1, 2))
    assert_(atleast_2d([[[3, 1], [4, 5]], [[3, 5], [1, 2]]]).shape == (2, 2, 2))