from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_to_dict_wide(self):
    df = DataFrame({f'A_{i:d}': [i] for i in range(256)})
    result = df.to_dict('records')[0]
    expected = {f'A_{i:d}': i for i in range(256)}
    assert result == expected