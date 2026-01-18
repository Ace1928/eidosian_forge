import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_series_non_zero_index(self):
    data = {0: {'id': 1, 'name': 'Foo', 'elements': {'a': 1}}, 1: {'id': 2, 'name': 'Bar', 'elements': {'b': 2}}, 2: {'id': 3, 'name': 'Baz', 'elements': {'c': 3}}}
    s = Series(data)
    s.index = [1, 2, 3]
    result = json_normalize(s)
    expected = DataFrame({'id': [1, 2, 3], 'name': ['Foo', 'Bar', 'Baz'], 'elements.a': [1.0, np.nan, np.nan], 'elements.b': [np.nan, 2.0, np.nan], 'elements.c': [np.nan, np.nan, 3.0]})
    tm.assert_frame_equal(result, expected)