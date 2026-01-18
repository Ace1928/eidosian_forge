from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize('orient,expected', [('list', {'A': [2, 5], 'B': [3, 6]}), ('dict', {'A': {0: 2, 1: 5}, 'B': {0: 3, 1: 6}})])
def test_to_dict_not_unique(self, orient, expected):
    df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'A', 'B'])
    result = df.to_dict(orient)
    assert result == expected