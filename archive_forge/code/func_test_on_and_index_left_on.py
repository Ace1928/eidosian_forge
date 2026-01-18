import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_on_and_index_left_on(self, trades, quotes):
    trades = trades.set_index('time')
    quotes = quotes.set_index('time')
    msg = 'Can only pass argument "left_on" OR "left_index" not both.'
    with pytest.raises(MergeError, match=msg):
        merge_asof(trades, quotes, left_on='price', left_index=True, right_index=True)