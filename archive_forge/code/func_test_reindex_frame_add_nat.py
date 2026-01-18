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
def test_reindex_frame_add_nat(self):
    rng = date_range('1/1/2000 00:00:00', periods=10, freq='10s')
    df = DataFrame({'A': np.random.default_rng(2).standard_normal(len(rng)), 'B': rng})
    result = df.reindex(range(15))
    assert np.issubdtype(result['B'].dtype, np.dtype('M8[ns]'))
    mask = isna(result)['B']
    assert mask[-5:].all()
    assert not mask[:-5].any()