from io import StringIO
import numpy as np
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from pandas import read_csv
class TestAnova2HC1(TestAnovaLM):

    def test_results(self):
        data = self.data.drop([0, 1, 2])
        anova_ii = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', data).fit()
        Sum_Sq = np.array([151.4065, 2.904723, 13.45718, 0.1905093, 27.60181])
        Df = np.array([1, 2, 2, 51])
        F = np.array([6.238771, 12.32983, 0.1529943, np.nan])
        PrF = np.array([0.01576555, 4.285456e-05, 0.858527, np.nan])
        results = anova_lm(anova_ii, typ='II', robust='hc1')
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['F'].values, F, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, PrF)