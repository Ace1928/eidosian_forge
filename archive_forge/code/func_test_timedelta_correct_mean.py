import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_timedelta_correct_mean(self):
    a = np.arange(1000, dtype='m8[s]')
    assert_array_equal(a.mean(), a.sum() / len(a))