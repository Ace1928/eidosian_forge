from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_array_as_cell_value(self):
    df = DataFrame(columns=['a', 'b'], dtype=object)
    df.loc[0] = {'a': np.zeros((2,)), 'b': np.zeros((2, 2))}
    expected = DataFrame({'a': [np.zeros((2,))], 'b': [np.zeros((2, 2))]})
    tm.assert_frame_equal(df, expected)