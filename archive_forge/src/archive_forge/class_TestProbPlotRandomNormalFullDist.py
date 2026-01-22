import numpy as np
import numpy.testing as nptest
from numpy.testing import assert_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics import gofplots
from statsmodels.graphics.gofplots import (
from statsmodels.graphics.utils import _import_mpl
class TestProbPlotRandomNormalFullDist(BaseProbplotMixin):

    def setup_method(self):
        np.random.seed(5)
        self.data = np.random.normal(loc=8.25, scale=3.25, size=37)
        self.prbplt = ProbPlot(self.data, dist=stats.norm(loc=8.5, scale=3.0))
        self.line = '45'
        super().setup_method()

    def test_loc_set(self):
        assert self.prbplt.loc == 8.5

    def test_scale_set(self):
        assert self.prbplt.scale == 3.0

    def test_exceptions(self):
        with pytest.raises(ValueError):
            ProbPlot(self.data, dist=stats.norm(loc=8.5, scale=3.0), fit=True)
        with pytest.raises(ValueError):
            ProbPlot(self.data, dist=stats.norm(loc=8.5, scale=3.0), distargs=(8.5, 3.0))
        with pytest.raises(ValueError):
            ProbPlot(self.data, dist=stats.norm(loc=8.5, scale=3.0), loc=8.5)
        with pytest.raises(ValueError):
            ProbPlot(self.data, dist=stats.norm(loc=8.5, scale=3.0), scale=3.0)