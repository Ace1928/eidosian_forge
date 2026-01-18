from statsmodels.compat.pandas import assert_frame_equal
from numpy.testing import (
import pandas as pd
from statsmodels.stats.anova import AnovaRM
def test_repeated_measures_aggregate_func():
    double_data = pd.concat([data, data], axis=0)
    assert_raises(ValueError, AnovaRM, double_data, 'DV', 'id', within=['A', 'B', 'D'])
    m1 = AnovaRM(double_data, 'DV', 'id', within=['A', 'B', 'D'], aggregate_func=pd.Series.mean)
    m2 = AnovaRM(double_data, 'DV', 'id', within=['A', 'B', 'D'], aggregate_func=pd.Series.median)
    assert_raises(AssertionError, assert_equal, m1.aggregate_func, m2.aggregate_func)
    assert_frame_equal(m1.fit().anova_table, m2.fit().anova_table)