from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_to_dict_numeric_names(self):
    df = DataFrame({str(i): [i] for i in range(5)})
    result = set(df.to_dict('records')[0].keys())
    expected = set(df.columns)
    assert result == expected