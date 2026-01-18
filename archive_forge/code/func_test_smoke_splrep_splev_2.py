import itertools
import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_,
from pytest import raises as assert_raises
import pytest
from scipy._lib._testutils import check_free_memory
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate._fitpack_py import (splrep, splev, bisplrep, bisplev,
from scipy.interpolate.dfitpack import regrid_smth
from scipy.interpolate._fitpack2 import dfitpack_int
@pytest.mark.parametrize('per', [0, 1])
@pytest.mark.parametrize('at_nodes', [True, False])
def test_smoke_splrep_splev_2(self, per, at_nodes):
    self.check_1(per=per, at_nodes=at_nodes)