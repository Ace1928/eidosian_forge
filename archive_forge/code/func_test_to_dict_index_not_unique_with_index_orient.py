from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_to_dict_index_not_unique_with_index_orient(self):
    df = DataFrame({'a': [1, 2], 'b': [0.5, 0.75]}, index=['A', 'A'])
    msg = "DataFrame index must be unique for orient='index'"
    with pytest.raises(ValueError, match=msg):
        df.to_dict(orient='index')