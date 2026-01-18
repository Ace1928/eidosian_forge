import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
def test_select_dtypes_typecodes(self):
    df = DataFrame(np.random.default_rng(2).random((5, 3)))
    FLOAT_TYPES = list(np.typecodes['AllFloat'])
    tm.assert_frame_equal(df.select_dtypes(FLOAT_TYPES), df)