import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import PeriodDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_setitem_raises_type():
    arr = PeriodArray(np.arange(3), dtype='period[D]')
    with pytest.raises(TypeError, match='int'):
        arr[0] = 1