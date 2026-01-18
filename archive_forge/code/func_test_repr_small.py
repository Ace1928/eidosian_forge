import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import PeriodDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_repr_small():
    arr = PeriodArray._from_sequence(['2000', '2001'], dtype='period[D]')
    result = str(arr)
    expected = "<PeriodArray>\n['2000-01-01', '2001-01-01']\nLength: 2, dtype: period[D]"
    assert result == expected