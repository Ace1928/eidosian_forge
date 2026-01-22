import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
class TestIsrealobj:

    def test_basic(self):
        z = np.array([-1, 0, 1])
        assert_(isrealobj(z))
        z = np.array([-1j, 0, -1])
        assert_(not isrealobj(z))