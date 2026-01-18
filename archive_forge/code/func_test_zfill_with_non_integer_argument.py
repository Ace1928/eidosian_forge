from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_zfill_with_non_integer_argument():
    value = Series(['-2', '+5'])
    wid = 'a'
    msg = f'width must be of integer type, not {type(wid).__name__}'
    with pytest.raises(TypeError, match=msg):
        value.str.zfill(wid)