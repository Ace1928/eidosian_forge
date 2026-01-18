import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
def test_select_dtypes_empty(self):
    df = DataFrame({'a': list('abc'), 'b': list(range(1, 4))})
    msg = 'at least one of include or exclude must be nonempty'
    with pytest.raises(ValueError, match=msg):
        df.select_dtypes()