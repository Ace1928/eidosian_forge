import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_timedelta_nat_format(self):
    assert_equal('NaT', '{0}'.format(np.timedelta64('nat')))