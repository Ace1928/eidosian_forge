from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('orient', ['dict', 'list', 'series', 'records', 'index'])
def test_to_dict_index_false_error(self, orient):
    df = DataFrame({'col1': [1, 2], 'col2': [3, 4]}, index=['row1', 'row2'])
    msg = "'index=False' is only valid when 'orient' is 'split' or 'tight'"
    with pytest.raises(ValueError, match=msg):
        df.to_dict(orient=orient, index=False)