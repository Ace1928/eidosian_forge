from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('df', [pd.DataFrame({'a': pd.to_datetime(['2017-01-22', '1970-01-01'])})])
def test_pos_raises(self, df):
    msg = "bad operand type for unary \\+: 'DatetimeArray'"
    with pytest.raises(TypeError, match=msg):
        +df
    with pytest.raises(TypeError, match=msg):
        +df['a']