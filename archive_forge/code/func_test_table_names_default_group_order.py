from statsmodels.compat.python import asbytes
from io import BytesIO
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_, assert_allclose, assert_almost_equal, assert_equal, \
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import (tukeyhsd, pairwise_tukeyhsd,
def test_table_names_default_group_order(self):
    t = self.res._results_table
    expected_order = [(b'medical', b'mental'), (b'medical', b'physical'), (b'mental', b'physical')]
    for i in range(1, 4):
        first_group = t[i][0].data
        second_group = t[i][1].data
        assert_((first_group, second_group) == expected_order[i - 1])