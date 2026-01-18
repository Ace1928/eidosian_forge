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
@pytest.mark.parametrize('kwargs, expected_axes_num, expected_layout, expected_shape', [({}, 1, (1, 1), (1,)), ({'layout': (3, 3)}, 1, (3, 3), (3, 3))])
def test_subplots_layout_single_column(self, kwargs, expected_axes_num, expected_layout, expected_shape):
    df = DataFrame(np.random.default_rng(2).random((10, 1)), index=list(string.ascii_letters[:10]))
    axes = df.plot(subplots=True, **kwargs)
    _check_axes_shape(axes, axes_num=expected_axes_num, layout=expected_layout)
    assert axes.shape == expected_shape