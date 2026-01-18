from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_td64_series_astype_object(self):
    tdser = Series(['59 Days', '59 Days', 'NaT'], dtype='timedelta64[ns]')
    result = tdser.astype(object)
    assert isinstance(result.iloc[0], timedelta)
    assert result.dtype == np.object_