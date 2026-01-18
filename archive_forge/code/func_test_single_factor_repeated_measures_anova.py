from statsmodels.compat.pandas import assert_frame_equal
from numpy.testing import (
import pandas as pd
from statsmodels.stats.anova import AnovaRM
def test_single_factor_repeated_measures_anova():
    """
    Testing single factor repeated measures anova
    Results reproduces R `ezANOVA` function from library ez
    """
    df = AnovaRM(data.iloc[:16, :], 'DV', 'id', within=['B']).fit()
    a = [[1, 7, 22.4, 0.002125452]]
    assert_array_almost_equal(df.anova_table.iloc[:, [1, 2, 0, 3]].values, a, decimal=5)