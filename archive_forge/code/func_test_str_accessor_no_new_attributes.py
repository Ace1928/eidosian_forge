from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_str_accessor_no_new_attributes(any_string_dtype):
    ser = Series(list('aabbcde'), dtype=any_string_dtype)
    with pytest.raises(AttributeError, match='You cannot add any new attribute'):
        ser.str.xlabel = 'a'