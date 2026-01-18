from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('val,exp_dtype', [('x', object), (5, IndexError), (1.1, object)])
def test_setitem_index_object(self, val, exp_dtype):
    obj = pd.Series([1, 2, 3, 4], index=pd.Index(list('abcd'), dtype=object))
    assert obj.index.dtype == object
    if exp_dtype is IndexError:
        temp = obj.copy()
        warn_msg = 'Series.__setitem__ treating keys as positions is deprecated'
        msg = 'index 5 is out of bounds for axis 0 with size 4'
        with pytest.raises(exp_dtype, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=warn_msg):
                temp[5] = 5
    else:
        exp_index = pd.Index(list('abcd') + [val], dtype=object)
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)