import pytest
import numpy as np
from numpy.testing import assert_allclose
import scipy.special as sc
class TestExpn:

    def test_out_of_domain(self):
        assert all(np.isnan([sc.expn(-1, 1.0), sc.expn(1, -1.0)]))