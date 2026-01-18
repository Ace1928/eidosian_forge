from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_partition_with_name(any_string_dtype):
    s = Series(['a,b', 'c,d'], name='xxx', dtype=any_string_dtype)
    result = s.str.partition(',')
    expected = DataFrame({0: ['a', 'c'], 1: [',', ','], 2: ['b', 'd']}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)