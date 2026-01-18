from datetime import datetime
from itertools import chain
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('index', [pd.timedelta_range(start=0, periods=2, freq='D'), [pd.Timedelta(days=1), pd.Timedelta(days=2)]])
def test_timedelta_index(self, index):
    xlims = (3, 1)
    ax = Series([1, 2], index=index).plot(xlim=xlims)
    assert ax.get_xlim() == (3, 1)