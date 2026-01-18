from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
def test_ser_mode_with_name(self):
    ser = Series([1, 1, 3], name='foo')
    result = ser.mode()
    expected = Series([1], name='foo')
    tm.assert_series_equal(result, expected)