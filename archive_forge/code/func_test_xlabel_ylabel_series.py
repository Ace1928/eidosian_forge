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
@pytest.mark.parametrize('index_name, old_label, new_label', [(None, '', 'new'), ('old', 'old', 'new'), (None, '', '')])
@pytest.mark.parametrize('kind', ['line', 'area', 'bar', 'barh', 'hist'])
def test_xlabel_ylabel_series(self, kind, index_name, old_label, new_label):
    ser = Series([1, 2, 3, 4])
    ser.index.name = index_name
    ax = ser.plot(kind=kind)
    if kind == 'barh':
        assert ax.get_xlabel() == ''
        assert ax.get_ylabel() == old_label
    elif kind == 'hist':
        assert ax.get_xlabel() == ''
        assert ax.get_ylabel() == 'Frequency'
    else:
        assert ax.get_ylabel() == ''
        assert ax.get_xlabel() == old_label
    ax = ser.plot(kind=kind, ylabel=new_label, xlabel=new_label)
    assert ax.get_ylabel() == new_label
    assert ax.get_xlabel() == new_label