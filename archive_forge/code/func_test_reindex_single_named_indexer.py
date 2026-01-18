from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_reindex_single_named_indexer(self):
    df = DataFrame({'A': [1, 2, 3], 'B': [1, 2, 3]})
    result = df.reindex([0, 1], columns=['A'])
    expected = DataFrame({'A': [1, 2]})
    tm.assert_frame_equal(result, expected)