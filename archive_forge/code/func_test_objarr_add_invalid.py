import datetime
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('op', [operator.add, ops.radd, operator.sub, ops.rsub])
def test_objarr_add_invalid(self, op, box_with_array):
    box = box_with_array
    obj_ser = Series(list('abc'), dtype=object, name='objects')
    obj_ser = tm.box_expected(obj_ser, box)
    msg = '|'.join(['can only concatenate str', 'unsupported operand type', 'must be str', 'has no kernel'])
    with pytest.raises(Exception, match=msg):
        op(obj_ser, 1)
    with pytest.raises(Exception, match=msg):
        op(obj_ser, np.array(1, dtype=np.int64))