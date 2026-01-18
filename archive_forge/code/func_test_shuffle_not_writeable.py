import warnings
import pytest
import numpy as np
from numpy.testing import (
from numpy import random
import sys
def test_shuffle_not_writeable(self):
    a = np.zeros(3)
    a.flags.writeable = False
    with pytest.raises(ValueError, match='read-only'):
        np.random.shuffle(a)