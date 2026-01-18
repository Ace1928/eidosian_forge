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
def test_q_fmt_lineoptions(self, close_figures):
    qqline(self.ax, 'q', dist=stats.norm, x=self.x, y=self.y, fmt=self.fmt, **self.lineoptions)