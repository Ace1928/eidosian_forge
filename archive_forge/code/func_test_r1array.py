import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
def test_r1array(self):
    """ Test to make sure equivalent Travis O's r1array function
        """
    assert_(atleast_1d(3).shape == (1,))
    assert_(atleast_1d(3j).shape == (1,))
    assert_(atleast_1d(3.0).shape == (1,))
    assert_(atleast_1d([[2, 3], [4, 5]]).shape == (2, 2))