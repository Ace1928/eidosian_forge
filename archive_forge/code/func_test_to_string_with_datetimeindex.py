from datetime import (
from io import StringIO
import re
import sys
from textwrap import dedent
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_to_string_with_datetimeindex(self):
    index = date_range('20130102', periods=6)
    ser = Series(1, index=index)
    result = ser.to_string()
    assert '2013-01-02' in result
    s2 = Series(2, index=[Timestamp('20130111'), NaT])
    ser = concat([s2, ser])
    result = ser.to_string()
    assert 'NaT' in result
    result = str(s2.index)
    assert 'NaT' in result