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
@pytest.mark.parametrize('layout, exp_layout', [[(2, 1), (2, 2)], [(2, -1), (2, 2)], [(-1, 2), (2, 2)]])
def test_subplots_multiple_axes_2_dim(self, layout, exp_layout):
    _, axes = mpl.pyplot.subplots(2, 2)
    df = DataFrame(np.random.default_rng(2).random((10, 4)), index=list(string.ascii_letters[:10]))
    with tm.assert_produces_warning(UserWarning):
        returned = df.plot(subplots=True, ax=axes, layout=layout, sharex=False, sharey=False)
        _check_axes_shape(returned, axes_num=4, layout=exp_layout)
        assert returned.shape == (4,)