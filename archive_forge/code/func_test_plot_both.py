import numpy as np
from numpy.testing import assert_equal, assert_raises
from pandas import Series
import pytest
from statsmodels.graphics.factorplots import _recode, interaction_plot
@pytest.mark.matplotlib
def test_plot_both(self, close_figures):
    fig = interaction_plot(self.weight, self.duration, self.days, colors=['red', 'blue'], markers=['D', '^'], ms=10)