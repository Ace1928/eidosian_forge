import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import pytest
from itertools import chain
def test_packbits_very_large():
    for s in range(950, 1050):
        for dt in '?bBhHiIlLqQ':
            x = np.ones((200, s), dtype=bool)
            np.packbits(x, axis=1)