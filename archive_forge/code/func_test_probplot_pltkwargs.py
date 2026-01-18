import numpy as np
import numpy.testing as nptest
from numpy.testing import assert_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics import gofplots
from statsmodels.graphics.gofplots import (
from statsmodels.graphics.utils import _import_mpl
@pytest.mark.matplotlib
def test_probplot_pltkwargs(self, close_figures):
    self.prbplt.probplot(ax=self.ax, line=self.line, marker='d', markerfacecolor='cornflowerblue', markeredgecolor='white', alpha=0.5)