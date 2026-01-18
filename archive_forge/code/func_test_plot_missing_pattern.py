import numpy as np
import pandas as pd
import pytest
from statsmodels.imputation import mice
import statsmodels.api as sm
from numpy.testing import assert_equal, assert_allclose
import warnings
@pytest.mark.matplotlib
def test_plot_missing_pattern(self, close_figures):
    df = gendat()
    imp_data = mice.MICEData(df)
    for row_order in ('pattern', 'raw'):
        for hide_complete_rows in (False, True):
            for color_row_patterns in (False, True):
                plt.clf()
                fig = imp_data.plot_missing_pattern(row_order=row_order, hide_complete_rows=hide_complete_rows, color_row_patterns=color_row_patterns)
                close_or_save(pdf, fig)
                close_figures()