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
@pytest.mark.filterwarnings('ignore:Series.__setitem__ treating keys as positions is deprecated:FutureWarning')
def test_setitem_ndarray_3d(self, index, frame_or_series, indexer_sli):
    obj = gen_obj(frame_or_series, index)
    idxr = indexer_sli(obj)
    nd3 = np.random.default_rng(2).integers(5, size=(2, 2, 2))
    if indexer_sli is tm.iloc:
        err = ValueError
        msg = f'Cannot set values with ndim > {obj.ndim}'
    else:
        err = ValueError
        msg = '|'.join(['Buffer has wrong number of dimensions \\(expected 1, got 3\\)', 'Cannot set values with ndim > 1', 'Index data must be 1-dimensional', 'Data must be 1-dimensional', 'Array conditional must be same shape as self'])
    with pytest.raises(err, match=msg):
        idxr[nd3] = 0