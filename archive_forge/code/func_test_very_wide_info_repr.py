from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_very_wide_info_repr(self):
    df = DataFrame(np.random.randn(10, 20), columns=tm.rands_array(10, 20))
    repr(df)