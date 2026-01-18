from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('orient, expected', [('dict', {'a': {0: 1, 1: None}}), ('list', {'a': [1, None]}), ('split', {'index': [0, 1], 'columns': ['a'], 'data': [[1], [None]]}), ('tight', {'index': [0, 1], 'columns': ['a'], 'data': [[1], [None]], 'index_names': [None], 'column_names': [None]}), ('records', [{'a': 1}, {'a': None}]), ('index', {0: {'a': 1}, 1: {'a': None}})])
def test_to_dict_na_to_none(self, orient, expected):
    df = DataFrame({'a': [1, NA]}, dtype='Int64')
    result = df.to_dict(orient=orient)
    assert result == expected