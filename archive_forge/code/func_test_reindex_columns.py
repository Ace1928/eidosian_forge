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
def test_reindex_columns(self, float_frame):
    new_frame = float_frame.reindex(columns=['A', 'B', 'E'])
    tm.assert_series_equal(new_frame['B'], float_frame['B'])
    assert np.isnan(new_frame['E']).all()
    assert 'C' not in new_frame
    new_frame = float_frame.reindex(columns=[])
    assert new_frame.empty