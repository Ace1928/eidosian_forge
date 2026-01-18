from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_NAT_with_None(self):
    df = DataFrame([pd.NaT, pd.NaT])
    result = df.replace({pd.NaT: None, np.nan: None})
    expected = DataFrame([None, None])
    tm.assert_frame_equal(result, expected)