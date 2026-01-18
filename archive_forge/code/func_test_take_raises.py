import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import PeriodDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_take_raises():
    arr = PeriodArray._from_sequence(['2000', '2001'], dtype='period[D]')
    with pytest.raises(IncompatibleFrequency, match='freq'):
        arr.take([0, -1], allow_fill=True, fill_value=pd.Period('2000', freq='W'))
    msg = "value should be a 'Period' or 'NaT'. Got 'str' instead"
    with pytest.raises(TypeError, match=msg):
        arr.take([0, -1], allow_fill=True, fill_value='foo')