from io import StringIO
import numpy as np
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from pandas import read_csv
class TestAnova3HC0(TestAnovaLM):

    def test_results(self):
        data = self.data.drop([0, 1, 2])
        anova_iii = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', data).fit()
        Sum_Sq = np.array([151.4065, 2.904723, 13.45718, 0.1905093, 27.60181])
        Df = np.array([1, 1, 2, 2, 51])
        F = np.array([298.3404, 5.723638, 13.76069, 0.1709936, np.nan])
        PrF = np.array([5.876255e-23, 0.02046031, 1.662826e-05, 0.8433081, np.nan])
        results = anova_lm(anova_iii, typ='III', robust='hc0')
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['F'].values, F, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, PrF)