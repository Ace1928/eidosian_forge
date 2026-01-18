import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_partial_string_timestamp_multiindex_str_key_raises(df):
    with pytest.raises(KeyError, match="'2016-01-01'"):
        df['2016-01-01']