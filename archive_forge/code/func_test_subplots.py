import string
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
@pytest.mark.parametrize('kind', ['bar', 'barh', 'line', 'area'])
def test_subplots(self, kind):
    df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
    axes = df.plot(kind=kind, subplots=True, sharex=True, legend=True)
    _check_axes_shape(axes, axes_num=3, layout=(3, 1))
    assert axes.shape == (3,)
    for ax, column in zip(axes, df.columns):
        _check_legend_labels(ax, labels=[pprint_thing(column)])
    for ax in axes[:-2]:
        _check_visible(ax.xaxis)
        _check_visible(ax.get_xticklabels(), visible=False)
        if kind != 'bar':
            _check_visible(ax.get_xticklabels(minor=True), visible=False)
        _check_visible(ax.xaxis.get_label(), visible=False)
        _check_visible(ax.get_yticklabels())
    _check_visible(axes[-1].xaxis)
    _check_visible(axes[-1].get_xticklabels())
    _check_visible(axes[-1].get_xticklabels(minor=True))
    _check_visible(axes[-1].xaxis.get_label())
    _check_visible(axes[-1].get_yticklabels())