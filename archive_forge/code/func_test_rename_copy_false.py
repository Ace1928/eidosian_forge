from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_copy_false(self, using_copy_on_write, warn_copy_on_write):
    ser = Series(['foo', 'bar'])
    ser_orig = ser.copy()
    shallow_copy = ser.rename({1: 9}, copy=False)
    with tm.assert_cow_warning(warn_copy_on_write):
        ser[0] = 'foobar'
    if using_copy_on_write:
        assert ser_orig[0] == shallow_copy[0]
        assert ser_orig[1] == shallow_copy[9]
    else:
        assert ser[0] == shallow_copy[0]
        assert ser[1] == shallow_copy[9]