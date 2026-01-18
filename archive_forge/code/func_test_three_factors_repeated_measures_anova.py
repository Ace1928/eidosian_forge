from statsmodels.compat.pandas import assert_frame_equal
from numpy.testing import (
import pandas as pd
from statsmodels.stats.anova import AnovaRM
def test_three_factors_repeated_measures_anova():
    """
    Testing three factors repeated measures anova
    Results reproduces R `ezANOVA` function from library ez
    """
    df = AnovaRM(data, 'DV', 'id', within=['A', 'B', 'D']).fit()
    a = [[1, 7, 8.7650709, 0.021087505], [2, 14, 8.4985785, 0.003833921], [1, 7, 20.5076546, 0.002704428], [2, 14, 0.8457797, 0.450021759], [1, 7, 21.7593382, 0.002301792], [2, 14, 6.2416695, 0.011536846], [2, 14, 5.4253359, 0.018010647]]
    assert_array_almost_equal(df.anova_table.iloc[:, [1, 2, 0, 3]].values, a, decimal=5)