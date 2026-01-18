import numpy as np
import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import (
import pandas._testing as tm
def test_td64_to_tdstruct():
    val = 12454636234
    res1 = py_td64_to_tdstruct(val, NpyDatetimeUnit.NPY_FR_ns.value)
    exp1 = {'days': 0, 'hrs': 0, 'min': 0, 'sec': 12, 'ms': 454, 'us': 636, 'ns': 234, 'seconds': 12, 'microseconds': 454636, 'nanoseconds': 234}
    assert res1 == exp1
    res2 = py_td64_to_tdstruct(val, NpyDatetimeUnit.NPY_FR_us.value)
    exp2 = {'days': 0, 'hrs': 3, 'min': 27, 'sec': 34, 'ms': 636, 'us': 234, 'ns': 0, 'seconds': 12454, 'microseconds': 636234, 'nanoseconds': 0}
    assert res2 == exp2
    res3 = py_td64_to_tdstruct(val, NpyDatetimeUnit.NPY_FR_ms.value)
    exp3 = {'days': 144, 'hrs': 3, 'min': 37, 'sec': 16, 'ms': 234, 'us': 0, 'ns': 0, 'seconds': 13036, 'microseconds': 234000, 'nanoseconds': 0}
    assert res3 == exp3
    res4 = py_td64_to_tdstruct(val, NpyDatetimeUnit.NPY_FR_s.value)
    exp4 = {'days': 144150, 'hrs': 21, 'min': 10, 'sec': 34, 'ms': 0, 'us': 0, 'ns': 0, 'seconds': 76234, 'microseconds': 0, 'nanoseconds': 0}
    assert res4 == exp4