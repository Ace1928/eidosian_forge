import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_with_duplicates(self, datapath, trades, quotes, asof):
    q = pd.concat([quotes, quotes]).sort_values(['time', 'ticker']).reset_index(drop=True)
    result = merge_asof(trades, q, on='time', by='ticker')
    expected = self.prep_data(asof)
    tm.assert_frame_equal(result, expected)