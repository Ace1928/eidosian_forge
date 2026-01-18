from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
def test_setitem_mask_categorical(self):
    cats2 = Categorical(['a', 'a', 'b', 'b', 'a', 'a', 'a'], categories=['a', 'b'])
    idx2 = Index(['h', 'i', 'j', 'k', 'l', 'm', 'n'])
    values2 = [1, 1, 2, 2, 1, 1, 1]
    exp_multi_row = DataFrame({'cats': cats2, 'values': values2}, index=idx2)
    catsf = Categorical(['a', 'a', 'c', 'c', 'a', 'a', 'a'], categories=['a', 'b', 'c'])
    idxf = Index(['h', 'i', 'j', 'k', 'l', 'm', 'n'])
    valuesf = [1, 1, 3, 3, 1, 1, 1]
    df = DataFrame({'cats': catsf, 'values': valuesf}, index=idxf)
    exp_fancy = exp_multi_row.copy()
    exp_fancy['cats'] = exp_fancy['cats'].cat.set_categories(['a', 'b', 'c'])
    mask = df['cats'] == 'c'
    df[mask] = ['b', 2]
    tm.assert_frame_equal(df, exp_fancy)