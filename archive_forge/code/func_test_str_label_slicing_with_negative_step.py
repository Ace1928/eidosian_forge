import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
@pytest.mark.parametrize('idx', [_mklbl('A', 20), np.arange(20) + 100, np.linspace(100, 150, 20)])
def test_str_label_slicing_with_negative_step(self, idx):
    SLC = pd.IndexSlice
    idx = Index(idx)
    ser = Series(np.arange(20), index=idx)
    tm.assert_indexing_slices_equivalent(ser, SLC[idx[9]::-1], SLC[9::-1])
    tm.assert_indexing_slices_equivalent(ser, SLC[:idx[9]:-1], SLC[:8:-1])
    tm.assert_indexing_slices_equivalent(ser, SLC[idx[13]:idx[9]:-1], SLC[13:8:-1])
    tm.assert_indexing_slices_equivalent(ser, SLC[idx[9]:idx[13]:-1], SLC[:0])