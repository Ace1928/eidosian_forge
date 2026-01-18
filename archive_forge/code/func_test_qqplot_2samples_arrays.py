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
def test_qqplot_2samples_arrays(self, close_figures):
    for line in ['r', 'q', '45', 's']:
        qqplot_2samples(self.res, self.other_array, line=line)