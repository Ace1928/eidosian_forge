from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_setitem_new_key_tz(self, indexer_sl):
    vals = [to_datetime(42).tz_localize('UTC'), to_datetime(666).tz_localize('UTC')]
    expected = Series(vals, index=Index(['foo', 'bar'], dtype=object))
    ser = Series(dtype=object)
    indexer_sl(ser)['foo'] = vals[0]
    indexer_sl(ser)['bar'] = vals[1]
    tm.assert_series_equal(ser, expected)