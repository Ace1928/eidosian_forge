from statsmodels.compat.python import lzip, lmap
from numpy.testing import (
import numpy as np
import pytest
from statsmodels.stats.libqsturng import qsturng, psturng
def test_v_equal_one(self):
    assert_almost_equal(0.1, psturng(0.2, 5, 1), 5)