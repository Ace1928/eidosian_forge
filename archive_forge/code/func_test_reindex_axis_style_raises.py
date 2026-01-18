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
def test_reindex_axis_style_raises(self):
    df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
        df.reindex([0, 1], columns=['A'], axis=1)
    with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
        df.reindex([0, 1], columns=['A'], axis='index')
    with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
        df.reindex(index=[0, 1], axis='index')
    with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
        df.reindex(index=[0, 1], axis='columns')
    with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
        df.reindex(columns=[0, 1], axis='columns')
    with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
        df.reindex(index=[0, 1], columns=[0, 1], axis='columns')
    with pytest.raises(TypeError, match='Cannot specify all'):
        df.reindex(labels=[0, 1], index=[0], columns=['A'])
    with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
        df.reindex(index=[0, 1], axis='index')
    with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
        df.reindex(index=[0, 1], axis='columns')
    with pytest.raises(TypeError, match='multiple values'):
        df.reindex([0, 1], labels=[0, 1])