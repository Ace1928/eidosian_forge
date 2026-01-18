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
def test_slice_with_zero_step_raises(self, index, indexer_sl, frame_or_series):
    obj = frame_or_series(np.arange(len(index)), index=index)
    with pytest.raises(ValueError, match='slice step cannot be zero'):
        indexer_sl(obj)[::0]