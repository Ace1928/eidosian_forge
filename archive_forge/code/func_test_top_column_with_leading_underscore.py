import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_top_column_with_leading_underscore(self):
    data = {'_id': {'a1': 10, 'l2': {'l3': 0}}, 'gg': 4}
    result = json_normalize(data, sep='_')
    expected = DataFrame([[4, 10, 0]], columns=['gg', '_id_a1', '_id_l2_l3'])
    tm.assert_frame_equal(result, expected)