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
def test_subplots_no_share_x(self, kind):
    df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
    axes = df.plot(kind=kind, subplots=True, sharex=False)
    for ax in axes:
        _check_visible(ax.xaxis)
        _check_visible(ax.get_xticklabels())
        _check_visible(ax.get_xticklabels(minor=True))
        _check_visible(ax.xaxis.get_label())
        _check_visible(ax.get_yticklabels())