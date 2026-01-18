from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('orient', ['dict', 'list', 'split', 'records', 'index', 'tight'])
@pytest.mark.parametrize('data,expected_types', (({'a': [np.int64(1), 1, np.int64(3)], 'b': [np.float64(1.0), 2.0, np.float64(3.0)], 'c': [np.float64(1.0), 2, np.int64(3)], 'd': [np.float64(1.0), 'a', np.int64(3)], 'e': [np.float64(1.0), ['a'], np.int64(3)], 'f': [np.float64(1.0), ('a',), np.int64(3)]}, {'a': [int, int, int], 'b': [float, float, float], 'c': [float, float, float], 'd': [float, str, int], 'e': [float, list, int], 'f': [float, tuple, int]}), ({'a': [1, 2, 3], 'b': [1.1, 2.2, 3.3]}, {'a': [int, int, int], 'b': [float, float, float]}), ({'a': [1, 'hello', 3], 'b': [1.1, 'world', 3.3]}, {'a': [int, str, int], 'b': [float, str, float]})))
def test_to_dict_returns_native_types(self, orient, data, expected_types):
    df = DataFrame(data)
    result = df.to_dict(orient)
    if orient == 'dict':
        assertion_iterator = ((i, key, value) for key, index_value_map in result.items() for i, value in index_value_map.items())
    elif orient == 'list':
        assertion_iterator = ((i, key, value) for key, values in result.items() for i, value in enumerate(values))
    elif orient in {'split', 'tight'}:
        assertion_iterator = ((i, key, result['data'][i][j]) for i in result['index'] for j, key in enumerate(result['columns']))
    elif orient == 'records':
        assertion_iterator = ((i, key, value) for i, record in enumerate(result) for key, value in record.items())
    elif orient == 'index':
        assertion_iterator = ((i, key, value) for i, record in result.items() for key, value in record.items())
    for i, key, value in assertion_iterator:
        assert value == data[key][i]
        assert type(value) is expected_types[key][i]