import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.parametrize('val1, val2', [(np.timedelta64(7, 'Y'), 15), (7.5, np.timedelta64(1, 'D'))])
def test_timedelta_modulus_type_resolution(self, val1, val2):
    with assert_raises_regex(TypeError, "'remainder' cannot use operands with types"):
        val1 % val2