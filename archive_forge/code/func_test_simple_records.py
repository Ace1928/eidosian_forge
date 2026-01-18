import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_simple_records(self):
    recs = [{'a': 1, 'b': 2, 'c': 3}, {'a': 4, 'b': 5, 'c': 6}, {'a': 7, 'b': 8, 'c': 9}, {'a': 10, 'b': 11, 'c': 12}]
    result = json_normalize(recs)
    expected = DataFrame(recs)
    tm.assert_frame_equal(result, expected)