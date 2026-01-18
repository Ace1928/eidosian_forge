from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('into, expected', [(dict, {0: {'int_col': 1, 'float_col': 1.0}, 1: {'int_col': 2, 'float_col': 2.0}, 2: {'int_col': 3, 'float_col': 3.0}}), (OrderedDict, OrderedDict([(0, {'int_col': 1, 'float_col': 1.0}), (1, {'int_col': 2, 'float_col': 2.0}), (2, {'int_col': 3, 'float_col': 3.0})])), (defaultdict(dict), defaultdict(dict, {0: {'int_col': 1, 'float_col': 1.0}, 1: {'int_col': 2, 'float_col': 2.0}, 2: {'int_col': 3, 'float_col': 3.0}}))])
def test_to_dict_index_dtypes(self, into, expected):
    df = DataFrame({'int_col': [1, 2, 3], 'float_col': [1.0, 2.0, 3.0]})
    result = df.to_dict(orient='index', into=into)
    cols = ['int_col', 'float_col']
    result = DataFrame.from_dict(result, orient='index')[cols]
    expected = DataFrame.from_dict(expected, orient='index')[cols]
    tm.assert_frame_equal(result, expected)