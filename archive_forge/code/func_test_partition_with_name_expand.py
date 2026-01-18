from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_partition_with_name_expand(any_string_dtype):
    s = Series(['a,b', 'c,d'], name='xxx', dtype=any_string_dtype)
    result = s.str.partition(',', expand=False)
    expected = Series([('a', ',', 'b'), ('c', ',', 'd')], name='xxx')
    tm.assert_series_equal(result, expected)