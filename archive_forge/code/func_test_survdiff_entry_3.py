import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def test_survdiff_entry_3():
    ti = np.r_[2, 1, 5, 8, 7, 8, 8, 9, 4, 9]
    st = np.r_[1, 1, 1, 1, 1, 0, 1, 0, 0, 0]
    gr = np.r_[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    entry = np.r_[1, 1, 2, 2, 3, 3, 2, 1, 2, 0]
    z, p = survdiff(ti, st, gr)
    assert_allclose(z, 6.9543024)
    assert_allclose(p, 0.008361789)
    z, p = survdiff(ti, st, gr, entry=entry)
    assert_allclose(z, 6.75082959)
    assert_allclose(p, 0.00937041)