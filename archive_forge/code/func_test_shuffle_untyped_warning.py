import warnings
import pytest
import numpy as np
from numpy.testing import (
from numpy import random
import sys
@pytest.mark.parametrize('random', [np.random, np.random.RandomState(), np.random.default_rng()])
def test_shuffle_untyped_warning(self, random):
    values = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    with pytest.warns(UserWarning, match="you are shuffling a 'dict' object") as rec:
        random.shuffle(values)
    assert 'test_random' in rec[0].filename