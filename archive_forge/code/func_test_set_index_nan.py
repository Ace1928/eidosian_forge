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
def test_set_index_nan(self):
    df = DataFrame({'PRuid': {17: 'nonQC', 18: 'nonQC', 19: 'nonQC', 20: '10', 21: '11', 22: '12', 23: '13', 24: '24', 25: '35', 26: '46', 27: '47', 28: '48', 29: '59', 30: '10'}, 'QC': {17: 0.0, 18: 0.0, 19: 0.0, 20: np.nan, 21: np.nan, 22: np.nan, 23: np.nan, 24: 1.0, 25: np.nan, 26: np.nan, 27: np.nan, 28: np.nan, 29: np.nan, 30: np.nan}, 'data': {17: 7.95449, 18: 8.014261, 19: 7.859152000000001, 20: 0.8614035, 21: 0.8785311, 22: 0.8427041999999999, 23: 0.785877, 24: 0.7306246, 25: 0.8166856, 26: 0.8192708000000001, 27: 0.8070501, 28: 0.8144024000000001, 29: 0.8014085, 30: 0.8130774000000001}, 'year': {17: 2006, 18: 2007, 19: 2008, 20: 1985, 21: 1985, 22: 1985, 23: 1985, 24: 1985, 25: 1985, 26: 1985, 27: 1985, 28: 1985, 29: 1985, 30: 1986}}).reset_index()
    result = df.set_index(['year', 'PRuid', 'QC']).reset_index().reindex(columns=df.columns)
    tm.assert_frame_equal(result, df)