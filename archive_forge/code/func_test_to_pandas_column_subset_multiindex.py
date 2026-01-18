import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
def test_to_pandas_column_subset_multiindex(self):
    df = pd.DataFrame({'first': list(range(5)), 'second': list(range(5)), 'value': np.arange(5)})
    table = pa.Table.from_pandas(df.set_index(['first', 'second']))
    subset = table.select(['first', 'value'])
    result = subset.to_pandas()
    expected = df[['first', 'value']].set_index('first')
    tm.assert_frame_equal(result, expected)