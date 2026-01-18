import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
def test_series_index_name(df):
    grouped = df.loc[:, ['C']].groupby(df['A'])
    result = grouped.agg(lambda x: x.mean())
    assert result.index.name == 'A'