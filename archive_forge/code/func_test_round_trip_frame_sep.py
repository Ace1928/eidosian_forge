from textwrap import dedent
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.clipboard import (
@pytest.mark.parametrize('sep', [None, '\t', ',', '|'])
@pytest.mark.parametrize('encoding', [None, 'UTF-8', 'utf-8', 'utf8'])
def test_round_trip_frame_sep(self, df, sep, encoding):
    df.to_clipboard(excel=None, sep=sep, encoding=encoding)
    result = read_clipboard(sep=sep or '\t', index_col=0, encoding=encoding)
    tm.assert_frame_equal(df, result)