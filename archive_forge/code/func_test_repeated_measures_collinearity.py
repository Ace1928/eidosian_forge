from statsmodels.compat.pandas import assert_frame_equal
from numpy.testing import (
import pandas as pd
from statsmodels.stats.anova import AnovaRM
def test_repeated_measures_collinearity():
    data1 = data.iloc[:48, :].copy()
    data1['E'] = data1['A']
    assert_raises(ValueError, AnovaRM, data1, 'DV', 'id', within=['A', 'E'])