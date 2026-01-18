import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('pass_axis', [False, True])
def test_scatter_matrix_axis(self, pass_axis):
    pytest.importorskip('scipy')
    scatter_matrix = plotting.scatter_matrix
    ax = None
    if pass_axis:
        _, ax = mpl.pyplot.subplots(3, 3)
    df = DataFrame(np.random.default_rng(2).standard_normal((100, 3)))
    with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
        axes = _check_plot_works(scatter_matrix, frame=df, range_padding=0.1, ax=ax)
    axes0_labels = axes[0][0].yaxis.get_majorticklabels()
    expected = ['-2', '0', '2']
    _check_text_labels(axes0_labels, expected)
    _check_ticks_props(axes, xlabelsize=8, xrot=90, ylabelsize=8, yrot=0)