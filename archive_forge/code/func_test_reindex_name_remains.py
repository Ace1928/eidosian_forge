from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_reindex_name_remains(self):
    s = Series(np.random.default_rng(2).random(10))
    df = DataFrame(s, index=np.arange(len(s)))
    i = Series(np.arange(10), name='iname')
    df = df.reindex(i)
    assert df.index.name == 'iname'
    df = df.reindex(Index(np.arange(10), name='tmpname'))
    assert df.index.name == 'tmpname'
    s = Series(np.random.default_rng(2).random(10))
    df = DataFrame(s.T, index=np.arange(len(s)))
    i = Series(np.arange(10), name='iname')
    df = df.reindex(columns=i)
    assert df.columns.name == 'iname'