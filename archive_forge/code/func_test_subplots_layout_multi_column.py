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
@pytest.mark.parametrize('layout, exp_layout', [[(2, 2), (2, 2)], [(-1, 2), (2, 2)], [(2, -1), (2, 2)], [(1, 4), (1, 4)], [(-1, 4), (1, 4)], [(4, -1), (4, 1)]])
def test_subplots_layout_multi_column(self, layout, exp_layout):
    df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
    axes = df.plot(subplots=True, layout=layout)
    _check_axes_shape(axes, axes_num=3, layout=exp_layout)
    assert axes.shape == exp_layout